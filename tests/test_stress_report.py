from pathlib import Path
import importlib.util

ROOT=Path(__file__).resolve().parents[1]

def test_stress_distinguishes_final_validity_from_optimizer_convergence():
    spec=importlib.util.spec_from_file_location('stress_tool',ROOT/'tools/stress.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result=module.stress(24)
    assert result['main_solver_attempts']==result['main_solver_converged_cases']+result['main_solver_nonconverged_cases']
    assert result['maximum_converged_qp_relative_gap']<=1e-8
    assert result['final_failures']==0
