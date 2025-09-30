"""
Script d'audit de code complet pour Neural Chat Engine.
"""
import subprocess
import os
import json
from typing import Dict, Any

class CodeAuditor:
    def __init__(self, src_path: str = "src", tests_path: str = "tests"):
        self.src_path = src_path
        self.tests_path = tests_path
        self.report: Dict[str, Any] = {}

    def analyze_code_quality(self):
        # Pylint
        pylint_result = subprocess.getoutput(f"pylint {self.src_path} --output-format=json")
        self.report['pylint'] = json.loads(pylint_result) if pylint_result else {}
        # Black
        black_result = subprocess.getoutput(f"black --check {self.src_path} {self.tests_path}")
        self.report['black'] = black_result
        # MyPy
        mypy_result = subprocess.getoutput(f"mypy {self.src_path}")
        self.report['mypy'] = mypy_result

    def check_security_issues(self):
        import json
        # Bandit
        bandit_result = subprocess.getoutput(f"bandit -r {self.src_path} -f json")
        try:
            self.report['bandit'] = json.loads(bandit_result)
        except Exception:
            self.report['bandit'] = {'raw_output': bandit_result}
        # Safety
        safety_result = subprocess.getoutput("safety check --json")
        try:
            self.report['safety'] = json.loads(safety_result)
        except Exception:
            self.report['safety'] = {'raw_output': safety_result}

    def measure_test_coverage(self):
        # Coverage.py
        subprocess.getoutput(f"coverage run -m pytest {self.tests_path}")
        coverage_result = subprocess.getoutput("coverage report -m --json")
        self.report['coverage'] = json.loads(coverage_result) if coverage_result else {}

    def analyze_dependencies(self):
        # pip-audit
        pip_audit_result = subprocess.getoutput("pip-audit -f json")
        self.report['pip_audit'] = json.loads(pip_audit_result) if pip_audit_result else {}

    def check_documentation(self):
        # Docstring coverage (pydocstyle)
        pydocstyle_result = subprocess.getoutput(f"pydocstyle {self.src_path}")
        self.report['docstring_coverage'] = pydocstyle_result

    def generate_audit_report(self, output_path: str = "audit_report.json"):
        with open(output_path, "w") as f:
            json.dump(self.report, f, indent=2)
        print(f"Rapport d'audit généré : {output_path}")
        return self.report

if __name__ == "__main__":
    auditor = CodeAuditor()
    auditor.analyze_code_quality()
    auditor.check_security_issues()
    auditor.measure_test_coverage()
    auditor.analyze_dependencies()
    auditor.check_documentation()
    auditor.generate_audit_report()
