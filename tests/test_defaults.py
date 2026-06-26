import pytest

from oxpy_utils.defaults.defaults import (
    DefaultInput, get_default_input, default_input_exist,
    IncompleteInputError, MissingParamError,
)


class TestDefaultInputExist:
    def test_known_inputs_exist(self):
        for name in ("cpu_MC_relax", "cpu_MD", "cpu_MD_relax",
                     "cuda_MD", "cuda_MD_relax", "ffs", "ffs_cuda", "vmmc"):
            assert default_input_exist(name), f"Expected default input '{name}' to exist"

    def test_nonexistent_returns_false(self):
        assert default_input_exist("definitely_not_a_real_input") is False


class TestDefaultInput:
    def test_loads_file_on_construction(self):
        d = DefaultInput("cpu_MC_relax")
        assert isinstance(d._input, dict)
        assert len(d._input) > 0

    def test_getitem_string_value(self):
        d = DefaultInput("cpu_MC_relax")
        assert d["backend"] == "CPU"

    def test_getitem_numeric_value_returns_string(self):
        d = DefaultInput("cpu_MC_relax")
        # steps = 5e3 in the file
        assert d["steps"] == "5000.0"

    def test_getitem_missing_key_raises(self):
        d = DefaultInput("cpu_MC_relax")
        with pytest.raises(KeyError):
            _ = d["nonexistent_key"]

    # --- get_dict ---

    def test_get_dict_raises_if_expressions_not_evaluated(self):
        # cpu_MC_relax has f(steps) expressions that need evaluate() first
        d = DefaultInput("cpu_MC_relax")
        with pytest.raises(IncompleteInputError):
            d.get_dict()

    def test_get_dict_no_expressions_succeeds_without_evaluate(self):
        # vmmc.json has no f() expressions
        d = DefaultInput("vmmc")
        result = d.get_dict()
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_get_dict_all_values_are_strings(self):
        d = DefaultInput("vmmc")
        for v in d.get_dict().values():
            assert isinstance(v, str)

    # --- evaluate ---

    def test_evaluate_uses_static_values_from_dict(self):
        d = DefaultInput("cpu_MC_relax")
        d.evaluate()  # steps=5e3 is a static key in the file
        result = d.get_dict()
        # print_conf_interval = steps / 10 = 5000.0 / 10 = 500.0
        assert result["print_conf_interval"] == "500.0"
        assert result["print_energy_every"] == "500.0"

    def test_evaluate_explicit_kwarg_overrides_dict_default(self):
        d = DefaultInput("cpu_MC_relax")
        d.evaluate(steps=1000)
        result = d.get_dict()
        # 1000 / 10 = 100.0
        assert result["print_conf_interval"] == "100.0"
        assert result["print_energy_every"] == "100.0"

    def test_evaluate_resolves_all_expressions(self):
        d = DefaultInput("cpu_MC_relax")
        d.evaluate()
        # After evaluate, get_dict must not raise
        result = d.get_dict()
        assert all(isinstance(v, str) for v in result.values())

    def test_evaluate_missing_kwarg_raises_missing_param_error(self):
        d = DefaultInput("cpu_MC_relax")
        # Inject an expression that depends on a key not in the file
        d._input["custom_expr"] = "f(undeclared_var) = undeclared_var * 2"
        with pytest.raises(MissingParamError) as exc_info:
            d.evaluate()
        assert exc_info.value.missing == "undeclared_var"

    # --- reset ---

    def test_reset_restores_unevaluated_state(self):
        d = DefaultInput("cpu_MC_relax")
        d.evaluate()
        d.reset()
        with pytest.raises(IncompleteInputError):
            d.get_dict()

    def test_reset_allows_re_evaluate_with_different_kwargs(self):
        d = DefaultInput("cpu_MC_relax")
        d.evaluate(steps=1000)
        assert d.get_dict()["print_conf_interval"] == "100.0"

        d.reset()
        d.evaluate(steps=2000)
        assert d.get_dict()["print_conf_interval"] == "200.0"


class TestGetDefaultInput:
    def test_factory_returns_default_input_instance(self):
        d = get_default_input("cpu_MC_relax")
        assert isinstance(d, DefaultInput)

    def test_factory_result_is_usable(self):
        d = get_default_input("vmmc")
        result = d.get_dict()
        assert "sim_type" in result


class TestIncompleteInputError:
    def test_stores_key(self):
        e = IncompleteInputError("my_key")
        assert e.objectionable_input_key == "my_key"

    def test_str_contains_key(self):
        e = IncompleteInputError("my_key")
        assert "my_key" in str(e)


class TestMissingParamError:
    def test_stores_fields(self):
        e = MissingParamError("output_key", "missing_arg", ["a", "b"])
        assert e.key == "output_key"
        assert e.missing == "missing_arg"
        assert e.provided_params == ["a", "b"]

    def test_str_contains_key_and_missing(self):
        e = MissingParamError("output_key", "missing_arg", ["a", "b"])
        s = str(e)
        assert "output_key" in s
        assert "missing_arg" in s