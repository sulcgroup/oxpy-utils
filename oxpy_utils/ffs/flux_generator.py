from typing import Any, Union

import logging
import random as rnd
import shutil, glob
from multiprocessing import Process
from pathlib import Path


from .base_flux_sampler import BaseFluxSampler, success_pattern, read_output
from ..oxdna_simulation import Simulation, find_top_file
from .ffs_interface import FFSInterface, Condition, write_order_params, order_params
from ..utils.order_parameter import OrderParameter

'''
Forward Flux sampling: Flux generator a-la-Tom (Ouldridge?)

Flavio
'''


class FFSFluxGenerator(BaseFluxSampler):
    """

    intrestingly, I don't think this setup is actually specific to seperations
    """
    order_params: list[OrderParameter]

    # ------------------------------ interfaces ---------------------------------

    lambda_n: FFSInterface # success infterface within the context of flux generation.
    # lambda_s = total successs of process we are sampling. this needs to be included as a "fail" condition for
    # the fluxer for thoroughness reasons but in most cases it's not possible to reach the
    # success interface without first crossing lambda_0
    lambda_s: FFSInterface

    # the "-1" interface is the point at or before the start of the ffs procedure
    # all time should be spentv in the state-space past this interface
    lambda_neg1: FFSInterface


    # ---- conditions------
    # condition where system either crosses the lambda_-1 interface going backwards or crosses the lambda_{0} interface
    # previously called "both" but renamed bc that's not what "both" means
    pass_or_fail: Condition
    # apart-forward. condition where the system crosses the lambda_{-1} interface going forward.
    apart_fw: Condition
    # condition where the system either fails (crosses lambda_-1 going backwards) or succeds
    # (all bonds totally dissociated)
    fail_or_success: Condition

    relax_steps: int
    # for flux generation, the interface we're trying to reach is the same as the success interface for this step
    lambda_plus1 = property(lambda self: self.lambda_n)

    def set_interfaces(self,
                       lambda_0: FFSInterface,
                       lambda_neg1: FFSInterface,
                       lambda_s: FFSInterface,
                       lambda_fail: Union[None, FFSInterface] = None):
        """"
        set interfaces
        Parameters:
            lambda_0: first interface of the process we're studying
            lambda_neg1: interface before the process we're studying, moving forward
            lambda_s: success interface for full process
            lambda_fail: interface before the process we're studying, moving backward. if unspecified default to ~lambda_neg1


        """
        if lambda_fail is None:
            # if we use lambda_neg1.flip as default we run risk of having a condition that immediately occurs
            lambda_fail = ~lambda_neg1

        self.lambda_fail = lambda_fail
        self.lambda_n = lambda_0
        self.lambda_neg1 = lambda_neg1
        self.lambda_s = lambda_s

        # condition where the system has either crossed the lambda_0 interface or gone back across the lambda_-1 interface
        self.pass_or_fail = Condition(
            "both",
            [lambda_0, lambda_fail],
            "or"
        )

        # crossing across lambda_-1 going forward
        self.apart_fw = Condition(
            "apart_fw",
            [lambda_neg1]
        )

        # has achieved complete success (process complete) or failure (moving back across lambda_-1
        self.fail_or_success = Condition("fail-or-success",
                                         [lambda_fail, lambda_s],
                                         "or")

        self.order_params = order_params(lambda_0, lambda_neg1, lambda_s, lambda_fail)
        self.relax_steps = int(1e5) # default value

    def get_success_confs(self):
        """
        get success confs from destination directory
        """
        confs = glob.glob(str(self.destination_directory / success_pattern))
        return confs

    def total_success_time(self):
        stime = 0.
        confs = self.get_success_confs()
        for conf in confs:
            with open(conf, 'r') as f:
                t = int(f.readline().split('=')[1])
                stime += t
        return len(confs) / stime


    def compute_flux(self):
        """
        compute flux by running simulations until we have the desired number of successes
        """
        return len(self.get_success_confs()) / self.total_success_time()


    def run(self):
        assert self.desired_success_count > 1
        # make flux directory
        processes = []
        main_log = self.loghandler.spinoff("main")
        if self.success_count.value >= self.desired_success_count:
            main_log.info(f"Found enough existing successes (found {self.success_count.value}, required {self.desired_success_count}). Exiting...")
        main_log.info(f"Main: STARTING new shooting for {self.desired_success_count}")
        top_file = find_top_file(self.source_directory)
        shutil.copy(top_file,
                    self.destination_directory)

        tp = Process(target=self.timer)
        tp.start()
        if self.ncpus > 1:
            # construct one process for each cpu
            for i in range(self.ncpus):
                p = Process(target=self.ffs_process, args=(i, self.loghandler.spinoff(f"Worker{i}")))
                processes.append(p)


            main_log.info("Main: Starting processes...")
            for p in processes:
                p.start()

            main_log.info("Main: waiting for processes to finish")
            for p in processes:
                p.join()
        else:
            # for debugging: allow serial processing
            self.ffs_process(0, self.loghandler.spinoff("worker"))

        main_log.info("Main: Terminating timer")
        tp.terminate()  # terminate timer

        # print >> sys.stderr, "nstarted: %d, nsuccesses: %d success_prob: %g" % (nstarted, nsuccesses, nsuccesses/float(nstarted))
        main_log.info("terminating processes")

        main_log.info(f"Main: nsuccesses: {self.success_count.value - self.initial_success_count} in this run")

        # final computation of the flux
        if len(self.get_success_confs()):
            flux = self.compute_flux()

            main_log.info(
                f"average number of timesteps taken to reach a success (including possibly previous"
                f" runs with the same pattern) (aka 1./flux): {1/flux}")
            main_log.info(f"initial flux (includes previous runs if they were there): {flux}")
        else:
            raise Exception("No confs generated!!!")
            # main_log.info("No confs generated!!!")

    # TODO: add param to write a message to the directory explaining what we're trying to do
    def make_ffs_simulation(self,
                            origin: Union[Simulation, Path],
                            sim_dir: Path,
                            seed: int,
                            other_inputs: dict[str, Any],
                            ffs_condition: Union[Condition, None] = None) -> Simulation:
        """

        """
        # todo: employ matt's defaults system when he writes it
        sim = Simulation(origin if isinstance(origin, Path) else origin.sim_dir, sim_dir)
        sim.build(clean_build='force')

        # set default input to flux
        sim.input.swap_default_input(self.default_input_name)
        # set seed
        sim.input["seed"] = seed
        # make sure source directory exists
        assert sim.file_dir.exists() # todo custom exception

        # set ffs condition if any
        if ffs_condition is not None:
            # write order parameters file
            write_order_params(sim.sim_dir / "op.txt", *ffs_condition.get_order_params())
            # write ffs condition file
            ffs_condition.write(sim.sim_dir)
            sim.input["ffs_file"] = ffs_condition.file_name()
            sim.input["order_parameters_file"] = "op.txt"

        if (sim.file_dir / "forces.json").is_file():
            sim.add_forces()

        # update additl input params
        sim.input.modify_input(other_inputs)
        # add our input file params
        sim.input.modify_input(self.input_file_params)
        if self.seq_dependant:
            sim.make_sequence_dependant()

        return sim

    def ffs_process(self, process_idx: int, plogger: logging.Logger):
        """
        this function does the work of running the initial flux simulation, identifying a
        success or a failure, and taking appropriate actions
        note: i'm somewhat concerned about thing being parallelized
        """
        # the seed is the index + initial seed, and the last_conf has an index as well
        seed = self.initial_seed + process_idx
        myrng = rnd.Random()
        myrng.seed(seed)

        sim_counter = 0  # process-specific sim counter
        # outer while loop
        while self.success_count.value < self.desired_success_count:
            # ----- step 1: initial relax ---------
            # do this every time w/ a random seed to make sure we have different starting points for our simulation
            plogger.info("equilibration started")
            eq_sim = self.make_ffs_simulation(self.source_directory,
                                              self.tld() / f"p{process_idx}/sim{sim_counter}",
                                              myrng.randint(1, int(5e6)), {
                                                  "sim_type": "MD",
                                                  "steps": self.relax_steps,
                                                  "refresh_vel": 1,
                                                  "print_energy_every": 1e2,
                                                  "restart_step_counter": 0
                                              })
            eq_sim_id = (self.name, process_idx, sim_counter)
            # if running as part of a larger program, log info
            if self.update_queue is not None:
                self.update_queue.put((
                    "flux",
                    process_idx,
                    sim_counter,
                    # can find other input file params in directory using this info
                    "equilibrate",
                    "origin"
                ))
            sim_counter += 1
            # do not use OxpyRun multiprocessing, since we're already in an mps thread
            eq_sim.oxpy_run.run(subprocess=False)
            if eq_sim.oxpy_run.error_message:
                raise Exception(eq_sim.oxpy_run.error_message)

            plogger.info("equilibrated")

            # ---------- run for a bit for some reason? ---------
            # i'm like 70% sure this is to make sure we don't start out past lambda_{-1}
            # being a bit specific
            plogger.info("Running initial simulation to start before λ_{-1}")
            init_sim = self.make_ffs_simulation(eq_sim, self.tld() / f"p{process_idx}/sim{sim_counter}",
                                                myrng.randint(1, 50000),
                                                {
                "refresh_vel": 0,
                "restart_step_counter": 1,
                "steps": 1e10
            }, self.fail_or_success)
            if self.update_queue is not None:
                self.update_queue.put((
                    "flux",
                    process_idx,
                    sim_counter,
                    "reset",
                    eq_sim_id
                ))
            sim_counter += 1
            # # run

            # tried adding observable but this didn't work
            # init_sim.add_observable(Observable.hb_list(f"{1e5}", "bonds", True))
            init_sim.oxpy_run.run(subprocess=False)
            if init_sim.oxpy_run.error_message:
                raise Exception(init_sim.oxpy_run.error_message)

            # grab ffs values
            op_values = read_output(init_sim)
            complete_success = self.lambda_s.test(op_values[self.lambda_s.op.name])

            # if the simumation fully dissociated, we need to start over b/c we can't get any confs to shoot with
            if complete_success:
                plogger.info(f"Sim {sim_counter} has reached a complete success, restarting")
                continue

            plogger.info(f"Sim {sim_counter}" + " reached Q_{-2}...")

            # now run simulations until done or something
            while self.success_count.value < self.desired_success_count:
                # ----- cross lambda_{-1} going forward -----------------------
                # construct new simulation from output of previous simulation
                source_node_id = (self.name, process_idx, sim_counter - 1)
                sim = self.make_ffs_simulation(self.tld() / f"p{process_idx}/sim{sim_counter - 1}",
                                               self.tld() / f"p{process_idx}/sim{sim_counter}", myrng.randint(1, 50000), {
                                                   'refresh_vel': 0,
                                                   'restart_step_counter': 0,
                                                   "steps": 2e10
                                               }, self.apart_fw)
                assert (sim.sim_dir / "op.txt").is_file(), f"File {str(sim.sim_dir / 'op.txt')} not created!!"
                # run
                sim.oxpy_run.run(subprocess=False)
                plogger.info("Worker %d: reached lambda_{-1} going forwards" % process_idx)
                if self.update_queue is not None:
                    self.update_queue.put((
                        "flux",
                        process_idx,
                        sim_counter,
                        "to_l-1_fwd",
                        source_node_id
                    ))

                # ------- flux sample -------------
                # continue running simulation until we either fail or hit the lambda_{0} interface
                source_node_id = (self.name, process_idx, sim_counter)
                sim_counter += 1
                sim = self.make_ffs_simulation(sim,
                                               self.tld() / f"p{process_idx}/sim{sim_counter}",
                                               myrng.randint(1, 50000),
                                               {
                    'refresh_vel': 0,
                    'restart_step_counter': 0,
                    "steps": 2e10
                }, self.pass_or_fail)
                if self.update_queue is not None:
                    self.update_queue.put((
                        "flux",
                        process_idx,
                        sim_counter,
                        "flux_fwd",
                        source_node_id
                    ))
                # run

                sim.oxpy_run.run(subprocess=False)

                op_values = read_output(sim)
                success = self.lambda_n.test(op_values[self.lambda_n.op.name])
                failure = self.lambda_fail.test(op_values[self.lambda_fail.op.name])
                # if we've had a success
                if success and not failure:
                    with self.success_lock:
                        # increment successes
                        self.success_count.value += 1
                        # copy last conf to working directory
                        shutil.copy(
                            f"{sim.sim_dir}/{sim.input.input_dict['lastconf_file']}",
                            self.destination_directory / f"{success_pattern.replace('*', str(self.success_count.value))}"
                        )
                        if self.update_queue:
                            self.update_queue.put((
                                "CPY_CONF",
                                self.name,
                                process_idx,
                                sim_counter,
                                self.success_count.value - 1 # zero indexed
                            ))
                            self.update_queue.put((
                                "flux_report",
                                process_idx,
                                sim_counter,
                                True
                            ))
                    source_node_id = (self.name, process_idx, sim_counter)
                    sim_counter += 1

                    plogger.info("Worker %d: crossed interface lambda_{0} going forwards: SUCCESS" % process_idx)

                    # ---------------- continue back across lambda_{0} ----------------------
                    # now that the simulation is past the lambda_{0} interface, we need to continue running it
                    # run until simulation is fully dissociate or have the @ least starting bond count
                    sim = self.make_ffs_simulation(sim, self.tld() / f"p{process_idx}/sim{sim_counter}", myrng.randint(1, 50000), {
                        'refresh_vel': 0,
                        'restart_step_counter': 1,
                        "steps": 2e10
                    }, self.fail_or_success)

                    if self.update_queue is not None:
                        self.update_queue.put((
                            "flux",
                            process_idx,
                            sim_counter,
                            "flux_back",
                            source_node_id
                        ))
                    sim_counter += 1
                    # run
                    eq_sim.oxpy_run.run(subprocess=False)

                    op_values = read_output(init_sim)
                    # complete_failure = lambda_f.test(op_values[lambda_f.op.name])
                    complete_success = self.lambda_s.test(op_values[self.lambda_s.op.name])

                    # did we fully dissociate? gotta start over them
                    if complete_success:
                        shutil.copy(f"{sim.sim_dir}/{sim.input.input_dict['lastconf_file']}",
                                    "full_success" + str(self.success_count.value))
                        plogger.info(f"Worker {process_idx} has reached a complete success: restarting from equilibration")
                        break  # this breakes the innermost while cycle, which will also start next iteration of main loop
                    else:  # ok we're back in our begin state, can continue from that
                        plogger.info("Worker %d: crossed interface lambda_{-1} going backwards after success" % process_idx)
                elif failure and not success: # slammed into the fail condition
                    plogger.info("Worker %d: crossed interface lambda_{-1} going backwards" % process_idx)
                    sim_counter += 1
                else:
                    raise Exception("fail condition" + str(self.lambda_fail) + " coexist with success condition " + str(self.lambda_n) )
        plogger.info(f"Achieved desired success count, process #{process_idx} terminating....")


