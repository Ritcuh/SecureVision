import pytest
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.nlp_model import (
    classify_code, analyze_code_vulnerabilities, 
    generate_security_score, get_remediation_suggestions
)
from app.logic_layer import detect_code_vulnerabilities, get_owasp_mapping

class TestNLPModel:
    """Test cases for NLP model functionality."""
    
    def test_classify_code_basic(self):
        """Test basic code classification."""
        # Safe code sample
        safe_code = """
        function greetUser(name) {
            return "Hello, " + name;
        }
        """
        
        label, score = classify_code(safe_code)
        assert label in ["SAFE", "VULNERABLE"], f"Unexpected label: {label}"
        assert 0.0 <= score <= 1.0, f"Score out of range: {score}"
    
    def test_classify_code_vulnerable(self):
        """Test classification of vulnerable code."""
        vulnerable_code = """
        <script>
            var userInput = document.getElementById('input').value;
            document.write(userInput);
        </script>
        """
        
        label, score = classify_code(vulnerable_code)
        assert label in ["SAFE", "VULNERABLE"], f"Unexpected label: {label}"
        assert 0.0 <= score <= 1.0, f"Score out of range: {score}"
    
    def test_analyze_code_vulnerabilities(self):
        """Test comprehensive vulnerability analysis."""
        test_code = """
        <form method="get" action="/login">
            <input type="text" name="username">
            <input type="password" name="password">
            <input type="submit" value="Login">
        </form>
        <script>
            var pass = "admin123";
            eval(userInput);
        </script>
        """
        
        result = analyze_code_vulnerabilities(test_code)
        
        # Check required fields
        assert 'label' in result
        assert 'confidence' in result
        assert 'risk_level' in result
        assert 'vulnerabilities' in result
        assert 'complexity_metrics' in result
        
        # Check data types
        assert isinstance(result['confidence'], float)
        assert 0.0 <= result['confidence'] <= 1.0
        assert result['risk_level'] in ['safe', 'warning', 'low', 'medium', 'high', 'critical']
        assert isinstance(result['vulnerabilities'], list)
        assert isinstance(result['complexity_metrics'], dict)
    
    def test_detect_sql_injection(self):
        """Test SQL injection detection."""
        sql_injection_code = """
        String query = "SELECT * FROM users WHERE username = '" + username + "'";
        Statement stmt = connection.createStatement();
        ResultSet rs = stmt.executeQuery(query);
        """
        
        vulnerabilities = detect_code_vulnerabilities(sql_injection_code)
        
        # Should detect SQL injection
        sql_vulns = [v for v in vulnerabilities if v['type'] == 'sql_injection']
        assert len(sql_vulns) > 0, "SQL injection not detected"
        
        # Check vulnerability structure
        if sql_vulns:
            vuln = sql_vulns[0]
            assert 'severity' in vuln
            assert 'description' in vuln
            assert 'owasp_category' in vuln
            assert vuln['severity'] in ['low', 'medium', 'high', 'critical']
    
    def test_detect_xss_vulnerability(self):
        """Test XSS vulnerability detection."""
        xss_code = """
        <script>
            var userComment = getUrlParameter('comment');
            document.getElementById('output').innerHTML = userComment;
        </script>
        """
        
        vulnerabilities = detect_code_vulnerabilities(xss_code)
        
        # Should detect XSS
        xss_vulns = [v for v in vulnerabilities if v['type'] == 'xss']
        assert len(xss_vulns) > 0, "XSS vulnerability not detected"
    
    def test_detect_insecure_authentication(self):
        """Test insecure authentication detection."""
        auth_code = """
        <form method="get" action="/login">
            <input type="text" name="username">
            <input type="password" name="password">
        </form>
        """
        
        vulnerabilities = detect_code_vulnerabilities(auth_code)
        
        # Should detect insecure authentication
        auth_vulns = [v for v in vulnerabilities if v['type'] == 'insecure_authentication']
        assert len(auth_vulns) > 0, "Insecure authentication not detected"
    
    def test_detect_csrf_vulnerability(self):
        """Test CSRF vulnerability detection."""
        csrf_code = """
        <form method="post" action="/transfer">
            <input type="text" name="amount">
            <input type="text" name="to_account">
            <input type="submit" value="Transfer">
        </form>
        """
        
        vulnerabilities = detect_code_vulnerabilities(csrf_code)
        
        # Should detect missing CSRF protection
        csrf_vulns = [v for v in vulnerabilities if v['type'] == 'csrf']
        assert len(csrf_vulns) > 0, "CSRF vulnerability not detected"
    
    def test_detect_sensitive_data_exposure(self):
        """Test sensitive data exposure detection."""
        sensitive_code = """
        var config = {
            api_key: "sk-1234567890abcdef",
            password: "admin123",
            secret: "my-secret-key"
        };
        """
        
        vulnerabilities = detect_code_vulnerabilities(sensitive_code)
        
        # Should detect sensitive data exposure
        sensitive_vulns = [v for v in vulnerabilities if v['type'] == 'sensitive_data_exposure']
        assert len(sensitive_vulns) > 0, "Sensitive data exposure not detected"
    
    def test_owasp_mapping(self):
        """Test OWASP category mapping."""
        mappings = {
            'sql_injection': 'A03_2021',
            'xss': 'A03_2021',
            'csrf': 'A01_2021',
            'insecure_authentication': 'A07_2021',
            'sensitive_data_exposure': 'A02_2021'
        }
        
        for vuln_type, expected_category in mappings.items():
            category = get_owasp_mapping(vuln_type)
            assert category == expected_category, f"Wrong OWASP mapping for {vuln_type}"
    
    def test_generate_security_score(self):
        """Test security score generation."""
        # Test with high-risk analysis
        high_risk_analysis = {
            'label': 'VULNERABLE',
            'confidence': 0.9,
            'risk_level': 'high',
            'vulnerabilities': [
                {'type': 'sql_injection', 'severity': 'high'},
                {'type': 'xss', 'severity': 'medium'}
            ],
            'complexity_metrics': {
                'cyclomatic_complexity': 15,
                'nesting_depth': 6,
                'security_hotspots': 5,
                'comment_ratio': 0.05
            },
            'total_vulnerabilities': 2,
            'severity_distribution': {'critical': 0, 'high': 1, 'medium': 1, 'low': 0}
        }
        
        score = generate_security_score(high_risk_analysis)
        assert 0.0 <= score <= 1.0, f"Score out of range: {score}"
        assert score > 0.5, "High-risk code should have high security score"
        
        # Test with low-risk analysis
        low_risk_analysis = {
            'label': 'SAFE',
            'confidence': 0.9,
            'risk_level': 'safe',
            'vulnerabilities': [],
            'complexity_metrics': {
                'cyclomatic_complexity': 3,
                'nesting_depth': 2,
                'security_hotspots': 0,
                'comment_ratio': 0.3
            },
            'total_vulnerabilities': 0,
            'severity_distribution': {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        }
        
        score = generate_security_score(low_risk_analysis)
        assert 0.0 <= score <= 1.0, f"Score out of range: {score}"
        assert score < 0.5, "Low-risk code should have low security score"
    
    def test_get_remediation_suggestions(self):
        """Test remediation suggestions generation."""
        vulnerabilities = [
            {'type': 'sql_injection', 'severity': 'high'},
            {'type': 'xss', 'severity': 'medium'},
            {'type': 'csrf', 'severity': 'medium'}
        ]
        
        suggestions = get_remediation_suggestions(vulnerabilities)
        
        assert isinstance(suggestions, list), "Suggestions should be a list"
        assert len(suggestions) > 0, "Should provide suggestions for vulnerabilities"
        
        # Check suggestion structure
        for suggestion in suggestions:
            assert 'title' in suggestion, "Suggestion should have title"
            assert 'suggestions' in suggestion, "Suggestion should have suggestions list"
            assert isinstance(suggestion['suggestions'], list), "Suggestions should be a list"
    
    def test_empty_code_analysis(self):
        """Test analysis of empty or minimal code."""
        empty_code = ""
        result = analyze_code_vulnerabilities(empty_code)
        
        assert 'label' in result
        assert 'confidence' in result
        assert isinstance(result['vulnerabilities'], list)
        
        minimal_code = "// Just a comment"
        result = analyze_code_vulnerabilities(minimal_code)
        
        assert 'label' in result
        assert 'confidence' in result
        assert isinstance(result['vulnerabilities'], list)
    
    def test_multiple_vulnerability_types(self):
        """Test detection of multiple vulnerability types in single code sample."""
        complex_code = """
        <form method="get" action="/login">
            <input type="password" name="password">
            <input type="submit" value="Login">
        </form>
        <script>
            var apiKey = "secret-key-123";
            var userInput = document.getElementById('input').value;
            document.write(userInput);
            
            var query = "SELECT * FROM users WHERE id = " + userId;
            eval(maliciousCode);
        </script>
        """
        
        vulnerabilities = detect_code_vulnerabilities(complex_code)
        
        # Should detect multiple types
        vuln_types = set(v['type'] for v in vulnerabilities)
        assert len(vuln_types) >= 2, "Should detect multiple vulnerability types"
        
        # Check for expected types
        expected_types = {'insecure_authentication', 'xss', 'sensitive_data_exposure', 'sql_injection'}
        found_types = vuln_types.intersection(expected_types)
        assert len(found_types) >= 2, f"Expected to find at least 2 types from {expected_types}, found {found_types}"

def test_classify_code():
    """Standalone test function for basic functionality."""
    sample = "<script>alert('hi');</script>"
    label, score = classify_code(sample)
    assert label in ["SAFE", "VULNERABLE"], "Unexpected label returned"
    assert 0.0 <= score <= 1.0, "Score out of range"

def test_vulnerability_detection():
    """Test basic vulnerability detection."""
    code_sample = """
    <form method="get" action="/login">
        <input type="password" name="pass">
    </form>
    """
    
    vulnerabilities = detect_code_vulnerabilities(code_sample)
    assert isinstance(vulnerabilities, list), "Should return a list"

if __name__ == "__main__":
    # Run basic tests
    test_classify_code()
    test_vulnerability_detection()
    
    # Run full test suite if pytest is available
    try:
        pytest.main([__file__, "-v"])
    except ImportError:
        print("Pytest not available, running basic tests only")
        print("All basic tests passed!")