import numpy as np
import re
from typing import Dict, List, Optional, Tuple

# OWASP Top 10 2021 mapping
OWASP_TOP_10 = {
    'A01_2021': 'Broken Access Control',
    'A02_2021': 'Cryptographic Failures',
    'A03_2021': 'Injection',
    'A04_2021': 'Insecure Design',
    'A05_2021': 'Security Misconfiguration',
    'A06_2021': 'Vulnerable and Outdated Components',
    'A07_2021': 'Identification and Authentication Failures',
    'A08_2021': 'Software and Data Integrity Failures',
    'A09_2021': 'Security Logging and Monitoring Failures',
    'A10_2021': 'Server-Side Request Forgery (SSRF)'
}

def score_risk(vision_result: Optional[Dict], nlp_result: Optional[Dict]) -> Dict:
    """
    Calculate combined risk score based on vision and NLP analysis results.
    
    Args:
        vision_result: Dictionary containing vision analysis results
        nlp_result: Dictionary containing NLP analysis results
    
    Returns:
        Dictionary with risk level, score, and details
    """
    vision_score = 0.0
    nlp_score = 0.0
    
    # Process vision analysis results
    if vision_result:
        vision_confidence = vision_result.get('confidence', 0.0)
        vision_risk = vision_result.get('risk_level', 'safe').lower()
        
        risk_multipliers = {'safe': 0.2, 'warning': 0.6, 'risky': 1.0}
        vision_score = vision_confidence * risk_multipliers.get(vision_risk, 0.5)
    
    # Process NLP analysis results
    if nlp_result:
        nlp_confidence = nlp_result.get('confidence', 0.0)
        vulnerabilities = nlp_result.get('vulnerabilities', [])
        
        # Base score from confidence
        nlp_score = nlp_confidence
        
        # Adjust score based on vulnerability severity
        severity_weights = {'critical': 1.0, 'high': 0.8, 'medium': 0.6, 'low': 0.3}
        
        if vulnerabilities:
            max_severity_weight = max([
                severity_weights.get(v.get('severity', '').lower(), 0.3) 
                for v in vulnerabilities
            ])
            nlp_score = max(nlp_score, max_severity_weight)
    
    # Calculate combined score with weighted average
    if vision_result and nlp_result:
        combined_score = (vision_score * 0.4 + nlp_score * 0.6)  # NLP weighted higher
    elif vision_result:
        combined_score = vision_score
    elif nlp_result:
        combined_score = nlp_score
    else:
        combined_score = 0.5  # Default moderate risk
    
    # Determine risk level
    if combined_score >= 0.8:
        risk_level = "Critical"
    elif combined_score >= 0.6:
        risk_level = "High"
    elif combined_score >= 0.4:
        risk_level = "Medium"
    elif combined_score >= 0.2:
        risk_level = "Low"
    else:
        risk_level = "Minimal"
    
    return {
        'level': risk_level,
        'score': combined_score,
        'vision_score': vision_score,
        'nlp_score': nlp_score,
        'details': {
            'vision_analysis': vision_result is not None,
            'nlp_analysis': nlp_result is not None,
            'vulnerability_count': len(nlp_result.get('vulnerabilities', [])) if nlp_result else 0
        }
    }

def get_owasp_mapping(vulnerability_type: str) -> str:
    """
    Map detected vulnerabilities to OWASP Top 10 categories.
    
    Args:
        vulnerability_type: Type of vulnerability detected
    
    Returns:
        OWASP category identifier
    """
    vulnerability_type = vulnerability_type.lower()
    
    mapping = {
        'sql_injection': 'A03_2021',
        'xss': 'A03_2021',
        'cross_site_scripting': 'A03_2021',
        'csrf': 'A01_2021',
        'insecure_authentication': 'A07_2021',
        'weak_password': 'A07_2021',
        'insecure_direct_object_reference': 'A01_2021',
        'security_misconfiguration': 'A05_2021',
        'sensitive_data_exposure': 'A02_2021',
        'insecure_cryptographic_storage': 'A02_2021',
        'insufficient_transport_layer_protection': 'A02_2021',
        'unvalidated_redirects': 'A01_2021',
        'insecure_deserialization': 'A08_2021',
        'using_known_vulnerable_components': 'A06_2021',
        'insufficient_logging': 'A09_2021',
        'ssrf': 'A10_2021'
    }
    
    return mapping.get(vulnerability_type, 'A04_2021')  # Default to Insecure Design

def detect_code_vulnerabilities(code: str) -> List[Dict]:
    """
    Detect specific vulnerabilities in code using pattern matching.
    
    Args:
        code: Source code to analyze
    
    Returns:
        List of detected vulnerabilities
    """
    vulnerabilities = []
    code_lower = code.lower()
    
    # SQL Injection patterns
    sql_patterns = [
        r'select\s+.*\s+from\s+.*\s+where\s+.*\+.*',
        r'query\s*=\s*["\'].*\+.*["\']',
        r'execute\(.*\+.*\)',
        r'mysql_query\(.*\+.*\)'
    ]
    
    for pattern in sql_patterns:
        if re.search(pattern, code_lower):
            vulnerabilities.append({
                'type': 'sql_injection',
                'severity': 'high',
                'description': 'Potential SQL injection vulnerability detected',
                'owasp_category': get_owasp_mapping('sql_injection'),
                'code_snippet': _extract_code_snippet(code, pattern)
            })
            break
    
    # XSS patterns
    xss_patterns = [
        r'document\.write\(.*\)',
        r'innerHTML\s*=\s*.*\+',
        r'outerHTML\s*=\s*.*\+',
        r'<script>.*</script>',
        r'eval\(.*\)'
    ]
    
    for pattern in xss_patterns:
        if re.search(pattern, code_lower):
            vulnerabilities.append({
                'type': 'xss',
                'severity': 'high',
                'description': 'Potential Cross-Site Scripting (XSS) vulnerability detected',
                'owasp_category': get_owasp_mapping('xss'),
                'code_snippet': _extract_code_snippet(code, pattern)
            })
            break
    
    # Insecure authentication patterns
    auth_patterns = [
        r'method\s*=\s*["\']get["\'].*password',
        r'password.*=.*request\.get',
        r'md5\(.*password.*\)',
        r'sha1\(.*password.*\)'
    ]
    
    for pattern in auth_patterns:
        if re.search(pattern, code_lower):
            vulnerabilities.append({
                'type': 'insecure_authentication',
                'severity': 'medium',
                'description': 'Insecure authentication implementation detected',
                'owasp_category': get_owasp_mapping('insecure_authentication'),
                'code_snippet': _extract_code_snippet(code, pattern)
            })
            break
    
    # CSRF patterns
    csrf_patterns = [
        r'<form(?!.*csrf).*method\s*=\s*["\']post["\']',
        r'$.post\((?!.*csrf)',
        r'fetch\(.*method\s*:\s*["\']post["\'](?!.*csrf)'
    ]
    
    for pattern in csrf_patterns:
        if re.search(pattern, code_lower):
            vulnerabilities.append({
                'type': 'csrf',
                'severity': 'medium',
                'description': 'Missing CSRF protection detected',
                'owasp_category': get_owasp_mapping('csrf'),
                'code_snippet': _extract_code_snippet(code, pattern)
            })
            break
    
    # Sensitive data exposure patterns
    sensitive_patterns = [
        r'password\s*=\s*["\'].*["\']',
        r'api_key\s*=\s*["\'].*["\']',
        r'secret\s*=\s*["\'].*["\']',
        r'token\s*=\s*["\'].*["\']'
    ]
    
    for pattern in sensitive_patterns:
        if re.search(pattern, code_lower):
            vulnerabilities.append({
                'type': 'sensitive_data_exposure',
                'severity': 'low',
                'description': 'Potential hardcoded sensitive data detected',
                'owasp_category': get_owasp_mapping('sensitive_data_exposure'),
                'code_snippet': _extract_code_snippet(code, pattern)
            })
            break
    
    return vulnerabilities

def _extract_code_snippet(code: str, pattern: str) -> str:
    """Extract a relevant code snippet around the matched pattern."""
    match = re.search(pattern, code.lower())
    if match:
        start = max(0, match.start() - 20)
        end = min(len(code), match.end() + 20)
        return code[start:end].strip()
    return "Pattern matched but snippet extraction failed"

def generate_recommendations(risk_assessment: Dict, nlp_result: Optional[Dict]) -> List[Dict]:
    """
    Generate security recommendations based on risk assessment.
    
    Args:
        risk_assessment: Risk assessment results
        nlp_result: NLP analysis results
    
    Returns:
        List of security recommendations
    """
    recommendations = []
    
    # General recommendations based on risk level
    risk_level = risk_assessment['level'].lower()
    
    if risk_level in ['critical', 'high']:
        recommendations.append({
            'title': 'Immediate Security Review Required',
            'description': 'High-risk vulnerabilities detected. Conduct immediate security review and implement fixes before deployment.',
            'priority': 'critical'
        })
    
    # Specific recommendations based on detected vulnerabilities
    if nlp_result and 'vulnerabilities' in nlp_result:
        vulnerability_types = set(v['type'] for v in nlp_result['vulnerabilities'])
        
        if 'sql_injection' in vulnerability_types:
            recommendations.append({
                'title': 'Implement Parameterized Queries',
                'description': 'Use prepared statements and parameterized queries to prevent SQL injection attacks. Avoid string concatenation in SQL queries.',
                'priority': 'high'
            })
        
        if 'xss' in vulnerability_types:
            recommendations.append({
                'title': 'Implement Output Encoding',
                'description': 'Encode all user input before displaying in HTML. Use context-appropriate encoding (HTML, JavaScript, CSS, URL).',
                'priority': 'high'
            })
        
        if 'csrf' in vulnerability_types:
            recommendations.append({
                'title': 'Add CSRF Protection',
                'description': 'Implement CSRF tokens in all state-changing operations. Use SameSite cookie attributes and validate referrer headers.',
                'priority': 'medium'
            })
        
        if 'insecure_authentication' in vulnerability_types:
            recommendations.append({
                'title': 'Strengthen Authentication',
                'description': 'Use HTTPS for authentication, implement strong password policies, and use secure hashing algorithms like bcrypt.',
                'priority': 'high'
            })
        
        if 'sensitive_data_exposure' in vulnerability_types:
            recommendations.append({
                'title': 'Secure Configuration Management',
                'description': 'Remove hardcoded secrets from source code. Use environment variables or secure configuration management systems.',
                'priority': 'medium'
            })
    
    # Add general security best practices
    recommendations.extend([
        {
            'title': 'Regular Security Audits',
            'description': 'Conduct regular security assessments and penetration testing to identify new vulnerabilities.',
            'priority': 'low'
        },
        {
            'title': 'Security Headers Implementation',
            'description': 'Implement security headers like Content-Security-Policy, X-Frame-Options, and X-XSS-Protection.',
            'priority': 'medium'
        },
        {
            'title': 'Dependency Security Scanning',
            'description': 'Regularly scan and update third-party dependencies to address known vulnerabilities.',
            'priority': 'medium'
        }
    ])
    
    return recommendations[:5]  # Return top 5 recommendations

def calculate_risk_trend(historical_scores: List[float]) -> Dict:
    """
    Calculate risk trend based on historical scores.
    
    Args:
        historical_scores: List of historical risk scores
    
    Returns:
        Dictionary with trend information
    """
    if len(historical_scores) < 2:
        return {'trend': 'insufficient_data', 'change': 0.0}
    
    recent_avg = np.mean(historical_scores[-3:])  # Last 3 scores
    older_avg = np.mean(historical_scores[:-3])   # All but last 3
    
    change = recent_avg - older_avg
    
    if change > 0.1:
        trend = 'increasing'
    elif change < -0.1:
        trend = 'decreasing'
    else:
        trend = 'stable'
    
    return {
        'trend': trend,
        'change': change,
        'recent_average': recent_avg,
        'historical_average': older_avg
    }
