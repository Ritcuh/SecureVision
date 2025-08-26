import re
import numpy as np
from typing import Dict, List, Tuple, Optional
from app.logic_layer import detect_code_vulnerabilities, get_owasp_mapping

# Mock transformer pipeline for environments without transformers
class MockCodeBERTClassifier:
    """Mock classifier that mimics CodeBERT behavior for demonstration."""
    
    def __init__(self):
        self.vulnerability_patterns = [
            'sql', 'injection', 'xss', 'csrf', 'authentication', 
            'password', 'script', 'alert', 'eval', 'innerHTML'
        ]
    
    def __call__(self, text: str) -> List[Dict]:
        # Simple heuristic-based classification
        text_lower = text.lower()
        
        vulnerability_score = 0.0
        found_patterns = []
        
        # Check for security-related patterns
        for pattern in self.vulnerability_patterns:
            if pattern in text_lower:
                vulnerability_score += 0.15
                found_patterns.append(pattern)
        
        # Additional specific checks
        if re.search(r'<script.*?>.*?</script>', text_lower):
            vulnerability_score += 0.3
            
        if re.search(r'select.*from.*where.*\+', text_lower):
            vulnerability_score += 0.4
            
        if 'method="get"' in text_lower and 'password' in text_lower:
            vulnerability_score += 0.35
            
        if re.search(r'eval\s*\(', text_lower):
            vulnerability_score += 0.25
        
        # Normalize score
        vulnerability_score = min(vulnerability_score, 1.0)
        safe_score = 1.0 - vulnerability_score
        
        # Return in format similar to transformers pipeline
        if vulnerability_score > safe_score:
            return [{'label': 'VULNERABLE', 'score': vulnerability_score}]
        else:
            return [{'label': 'SAFE', 'score': safe_score}]

def get_code_analyzer():
    """
    Get code analyzer. Falls back to mock implementation if transformers unavailable.
    """
    try:
        from transformers import pipeline
        # Try to load CodeBERT or similar model
        classifier = pipeline(
            "text-classification", 
            model="microsoft/codebert-base",
            return_all_scores=True
        )
        return classifier
    except ImportError:
        print("Transformers not available, using mock classifier")
        return MockCodeBERTClassifier()
    except Exception as e:
        print(f"Error loading transformer model: {e}, using mock classifier")
        return MockCodeBERTClassifier()

def classify_code(code_snippet: str) -> Tuple[str, float]:
    """
    Classify code snippet for security vulnerabilities.
    
    Args:
        code_snippet: Source code to analyze
    
    Returns:
        Tuple of (label, confidence_score)
    """
    analyzer = get_code_analyzer()
    result = analyzer(code_snippet)
    
    if isinstance(result, list) and len(result) > 0:
        if isinstance(result[0], dict):
            label = result[0].get('label', 'UNKNOWN')
            score = result[0].get('score', 0.5)
        else:
            label = 'SAFE'
            score = 0.5
    else:
        label = 'SAFE'
        score = 0.5
    
    return label, score

def analyze_code_vulnerabilities(code_snippet: str) -> Dict:
    """
    Comprehensive analysis of code vulnerabilities.
    
    Args:
        code_snippet: Source code to analyze
    
    Returns:
        Dictionary containing detailed vulnerability analysis
    """
    # Get basic classification
    label, confidence = classify_code(code_snippet)
    
    # Detect specific vulnerabilities
    vulnerabilities = detect_code_vulnerabilities(code_snippet)
    
    # Analyze code complexity and patterns
    complexity_metrics = analyze_code_complexity(code_snippet)
    
    # Determine overall risk level
    if vulnerabilities:
        severity_levels = [v['severity'] for v in vulnerabilities]
        if 'critical' in severity_levels:
            risk_level = 'critical'
        elif 'high' in severity_levels:
            risk_level = 'high'
        elif 'medium' in severity_levels:
            risk_level = 'medium'
        else:
            risk_level = 'low'
    else:
        risk_level = 'safe' if label == 'SAFE' else 'warning'
    
    # Adjust confidence based on number of vulnerabilities
    if vulnerabilities:
        # Higher confidence when specific vulnerabilities are detected
        confidence = min(confidence + (len(vulnerabilities) * 0.1), 1.0)
    
    return {
        'label': label,
        'confidence': confidence,
        'risk_level': risk_level,
        'vulnerabilities': vulnerabilities,
        'complexity_metrics': complexity_metrics,
        'total_vulnerabilities': len(vulnerabilities),
        'severity_distribution': _get_severity_distribution(vulnerabilities)
    }

def analyze_code_complexity(code: str) -> Dict:
    """
    Analyze code complexity metrics that may indicate security risks.
    
    Args:
        code: Source code to analyze
    
    Returns:
        Dictionary with complexity metrics
    """
    lines = code.split('\n')
    
    metrics = {
        'lines_of_code': len([line for line in lines if line.strip()]),
        'cyclomatic_complexity': estimate_cyclomatic_complexity(code),
        'nesting_depth': calculate_max_nesting_depth(code),
        'function_count': count_functions(code),
        'comment_ratio': calculate_comment_ratio(code),
        'security_hotspots': count_security_hotspots(code)
    }
    
    return metrics

def estimate_cyclomatic_complexity(code: str) -> int:
    """Estimate cyclomatic complexity by counting decision points."""
    decision_keywords = ['if', 'elif', 'else', 'while', 'for', 'try', 'except', 'case', 'switch']
    complexity = 1  # Base complexity
    
    code_lower = code.lower()
    for keyword in decision_keywords:
        complexity += len(re.findall(rf'\b{keyword}\b', code_lower))
    
    return complexity

def calculate_max_nesting_depth(code: str) -> int:
    """Calculate maximum nesting depth in code."""
    lines = code.split('\n')
    max_depth = 0
    current_depth = 0
    
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
            
        # Count opening braces/keywords that increase nesting
        opening_patterns = ['{', 'if ', 'for ', 'while ', 'try:', 'def ', 'class ']
        closing_patterns = ['}', 'end', 'except:', 'finally:']
        
        for pattern in opening_patterns:
            if pattern in line.lower():
                current_depth += 1
                break
        
        max_depth = max(max_depth, current_depth)
        
        for pattern in closing_patterns:
            if pattern in line.lower():
                current_depth = max(0, current_depth - 1)
                break
    
    return max_depth

def count_functions(code: str) -> int:
    """Count the number of functions in code."""
    function_patterns = [
        r'\bdef\s+\w+\s*\(',  # Python
        r'\bfunction\s+\w+\s*\(',  # JavaScript
        r'\b\w+\s*\([^)]*\)\s*{',  # C/C++/Java
        r'<\?php.*function\s+\w+',  # PHP
    ]
    
    count = 0
    for pattern in function_patterns:
        count += len(re.findall(pattern, code, re.IGNORECASE))
    
    return count

def calculate_comment_ratio(code: str) -> float:
    """Calculate ratio of comment lines to total lines."""
    lines = code.split('\n')
    total_lines = len([line for line in lines if line.strip()])
    
    if total_lines == 0:
        return 0.0
    
    comment_patterns = [r'^\s*#', r'^\s*//', r'^\s*/\*', r'^\s*\*', r'^\s*<!--']
    comment_lines = 0
    
    for line in lines:
        for pattern in comment_patterns:
            if re.match(pattern, line):
                comment_lines += 1
                break
    
    return comment_lines / total_lines if total_lines > 0 else 0.0

def count_security_hotspots(code: str) -> int:
    """Count potential security hotspots in code."""
    hotspot_patterns = [
        r'\beval\s*\(',
        r'\bexec\s*\(',
        r'\bsystem\s*\(',
        r'\bshell_exec\s*\(',
        r'innerHTML\s*=',
        r'document\.write\s*\(',
        r'mysql_query\s*\(',
        r'password\s*=\s*["\']',
        r'secret\s*=\s*["\']',
        r'api_key\s*=\s*["\']',
        r'<script[^>]*>',
        r'javascript:',
        r'onclick\s*=',
        r'onload\s*=',
        r'setTimeout\s*\(',
        r'setInterval\s*\('
    ]
    
    count = 0
    code_lower = code.lower()
    for pattern in hotspot_patterns:
        count += len(re.findall(pattern, code_lower))
    
    return count

def _get_severity_distribution(vulnerabilities: List[Dict]) -> Dict:
    """Get distribution of vulnerability severities."""
    distribution = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
    
    for vuln in vulnerabilities:
        severity = vuln.get('severity', 'low').lower()
        if severity in distribution:
            distribution[severity] += 1
    
    return distribution

def generate_security_score(analysis_result: Dict) -> float:
    """
    Generate a normalized security score based on analysis results.
    
    Args:
        analysis_result: Results from analyze_code_vulnerabilities
    
    Returns:
        Security score between 0 (secure) and 1 (insecure)
    """
    base_score = 1.0 - analysis_result['confidence'] if analysis_result['label'] == 'SAFE' else analysis_result['confidence']
    
    # Adjust based on vulnerability count and severity
    vuln_count = analysis_result['total_vulnerabilities']
    severity_dist = analysis_result['severity_distribution']
    
    severity_penalty = (
        severity_dist['critical'] * 0.25 +
        severity_dist['high'] * 0.15 +
        severity_dist['medium'] * 0.1 +
        severity_dist['low'] * 0.05
    )
    
    # Adjust based on complexity metrics
    complexity = analysis_result['complexity_metrics']
    complexity_penalty = 0
    
    if complexity['cyclomatic_complexity'] > 10:
        complexity_penalty += 0.1
    if complexity['nesting_depth'] > 5:
        complexity_penalty += 0.05
    if complexity['security_hotspots'] > 3:
        complexity_penalty += 0.1
    if complexity['comment_ratio'] < 0.1:
        complexity_penalty += 0.05
    
    final_score = min(base_score + severity_penalty + complexity_penalty, 1.0)
    return final_score

def get_remediation_suggestions(vulnerabilities: List[Dict]) -> List[Dict]:
    """
    Get specific remediation suggestions for detected vulnerabilities.
    
    Args:
        vulnerabilities: List of detected vulnerabilities
    
    Returns:
        List of remediation suggestions
    """
    suggestions = []
    
    vuln_types = set(v['type'] for v in vulnerabilities)
    
    remediation_map = {
        'sql_injection': {
            'title': 'SQL Injection Prevention',
            'suggestions': [
                'Use parameterized queries or prepared statements',
                'Implement input validation and sanitization',
                'Use stored procedures with parameters',
                'Apply principle of least privilege to database accounts'
            ]
        },
        'xss': {
            'title': 'Cross-Site Scripting (XSS) Prevention',
            'suggestions': [
                'Encode output data based on context (HTML, JavaScript, CSS)',
                'Use Content Security Policy (CSP) headers',
                'Validate and sanitize all user inputs',
                'Use secure coding frameworks with built-in XSS protection'
            ]
        },
        'csrf': {
            'title': 'Cross-Site Request Forgery (CSRF) Prevention',
            'suggestions': [
                'Implement CSRF tokens in all state-changing operations',
                'Use SameSite cookie attributes',
                'Validate HTTP Referer header',
                'Implement proper session management'
            ]
        },
        'insecure_authentication': {
            'title': 'Authentication Security',
            'suggestions': [
                'Use HTTPS for all authentication operations',
                'Implement strong password policies',
                'Use secure hashing algorithms (bcrypt, Argon2)',
                'Implement multi-factor authentication'
            ]
        },
        'sensitive_data_exposure': {
            'title': 'Sensitive Data Protection',
            'suggestions': [
                'Remove hardcoded secrets from source code',
                'Use environment variables for configuration',
                'Implement proper encryption for sensitive data',
                'Use secure configuration management systems'
            ]
        }
    }
    
    for vuln_type in vuln_types:
        if vuln_type in remediation_map:
            suggestions.append(remediation_map[vuln_type])
    
    return suggestions