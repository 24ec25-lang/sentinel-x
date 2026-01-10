"""
FastAPI implementation for threat detection endpoints.
Sentinel-X threat detection and analysis API.
"""

from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Sentinel-X Threat Detection API",
    description="Advanced threat detection and analysis platform",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== Enums ====================

class ThreatLevel(str, Enum):
    """Threat severity levels"""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


class ThreatType(str, Enum):
    """Types of threats"""
    MALWARE = "MALWARE"
    PHISHING = "PHISHING"
    DDoS = "DDoS"
    SQL_INJECTION = "SQL_INJECTION"
    XSS = "XSS"
    RANSOMWARE = "RANSOMWARE"
    ZERO_DAY = "ZERO_DAY"
    SUSPICIOUS_ACTIVITY = "SUSPICIOUS_ACTIVITY"
    ANOMALY = "ANOMALY"
    UNKNOWN = "UNKNOWN"


class AnalysisStatus(str, Enum):
    """Status of threat analysis"""
    PENDING = "PENDING"
    ANALYZING = "ANALYZING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


# ==================== Pydantic Models ====================

class ThreatIndicator(BaseModel):
    """Model for threat indicators"""
    name: str = Field(..., description="Name of the indicator")
    value: str = Field(..., description="Indicator value (IP, hash, domain, etc.)")
    indicator_type: str = Field(..., description="Type of indicator (IP, hash, domain, etc.)")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence score")


class ThreadAnalysisPayload(BaseModel):
    """Model for threat analysis request"""
    sample: str = Field(..., description="File hash, URL, IP address, or artifact to analyze")
    sample_type: str = Field(
        default="hash",
        description="Type of sample: hash, url, ip, or file"
    )
    priority: Optional[ThreatLevel] = Field(default=ThreatLevel.MEDIUM)
    tags: Optional[List[str]] = Field(default=[], description="Custom tags for the threat")


class ThreatDetection(BaseModel):
    """Model for threat detection response"""
    threat_id: str = Field(..., description="Unique threat identifier")
    threat_type: ThreatType = Field(..., description="Type of threat detected")
    threat_level: ThreatLevel = Field(..., description="Severity level")
    detected_at: datetime = Field(..., description="Detection timestamp")
    description: str = Field(..., description="Threat description")
    indicators: List[ThreatIndicator] = Field(default=[], description="Associated indicators")
    affected_systems: List[str] = Field(default=[], description="Affected systems/hosts")
    recommendations: List[str] = Field(default=[], description="Recommended actions")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Detection confidence")


class ThreatAnalysisResult(BaseModel):
    """Model for threat analysis result"""
    analysis_id: str = Field(..., description="Unique analysis identifier")
    status: AnalysisStatus = Field(..., description="Analysis status")
    sample: str = Field(..., description="Analyzed sample")
    threat_detected: bool = Field(..., description="Whether a threat was detected")
    threat: Optional[ThreatDetection] = None
    indicators: List[ThreatIndicator] = Field(default=[], description="Extracted indicators")
    analysis_details: Dict[str, Any] = Field(default={}, description="Detailed analysis information")
    analyzed_at: datetime = Field(default_factory=datetime.utcnow)
    processing_time_ms: float = Field(default=0.0, description="Processing time in milliseconds")


class ThreatReport(BaseModel):
    """Model for threat report"""
    report_id: str = Field(..., description="Unique report identifier")
    title: str = Field(..., description="Report title")
    threats: List[ThreatDetection] = Field(..., description="List of detected threats")
    total_threats: int = Field(..., description="Total number of threats")
    critical_count: int = Field(default=0, description="Count of critical threats")
    high_count: int = Field(default=0, description="Count of high severity threats")
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    summary: str = Field(..., description="Executive summary")
    recommendations: List[str] = Field(default=[], description="Overall recommendations")


class SecurityEvent(BaseModel):
    """Model for security events"""
    event_id: str = Field(..., description="Unique event identifier")
    event_type: str = Field(..., description="Type of security event")
    severity: ThreatLevel = Field(..., description="Event severity")
    source: str = Field(..., description="Event source (IP, host, etc.)")
    description: str = Field(..., description="Event description")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default={}, description="Additional metadata")


# ==================== Health Check ====================

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "Sentinel-X Threat Detection API"
    }


# ==================== Threat Detection Endpoints ====================

@app.post(
    "/api/v1/threats/detect",
    response_model=ThreatDetection,
    tags=["Threat Detection"],
    summary="Detect threats in provided artifacts"
)
async def detect_threat(
    payload: ThreadAnalysisPayload = Body(...)
) -> ThreatDetection:
    """
    Analyze provided artifacts for threats.
    
    Supported sample types:
    - hash: File hash (MD5, SHA1, SHA256)
    - url: Website or endpoint URL
    - ip: IPv4 or IPv6 address
    - file: Base64 encoded file content
    
    Returns detailed threat information if detected.
    """
    try:
        logger.info(f"Processing threat detection for sample: {payload.sample[:20]}...")
        
        # Mock threat detection logic
        threat_detected = _analyze_sample(
            payload.sample,
            payload.sample_type
        )
        
        if threat_detected:
            threat = ThreatDetection(
                threat_id=f"THR-{datetime.utcnow().timestamp()}",
                threat_type=threat_detected["type"],
                threat_level=threat_detected["level"],
                detected_at=datetime.utcnow(),
                description=threat_detected["description"],
                indicators=threat_detected.get("indicators", []),
                affected_systems=threat_detected.get("systems", []),
                recommendations=threat_detected.get("recommendations", []),
                confidence=threat_detected.get("confidence", 0.95)
            )
            logger.info(f"Threat detected: {threat.threat_id}")
            return threat
        else:
            raise HTTPException(
                status_code=404,
                detail="No threat detected for the provided sample"
            )
    except Exception as e:
        logger.error(f"Error in threat detection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post(
    "/api/v1/threats/analyze",
    response_model=ThreatAnalysisResult,
    tags=["Threat Analysis"],
    summary="Perform detailed threat analysis"
)
async def analyze_threat(
    payload: ThreadAnalysisPayload = Body(...)
) -> ThreatAnalysisResult:
    """
    Perform comprehensive threat analysis on artifacts.
    
    Includes:
    - Threat identification
    - Indicator extraction
    - Behavioral analysis
    - Detailed threat classification
    """
    try:
        logger.info(f"Starting detailed analysis for: {payload.sample[:20]}...")
        
        analysis_id = f"ANL-{datetime.utcnow().timestamp()}"
        threat_detected = _analyze_sample(
            payload.sample,
            payload.sample_type
        )
        
        if threat_detected:
            threat = ThreatDetection(
                threat_id=f"THR-{datetime.utcnow().timestamp()}",
                threat_type=threat_detected["type"],
                threat_level=threat_detected["level"],
                detected_at=datetime.utcnow(),
                description=threat_detected["description"],
                indicators=threat_detected.get("indicators", []),
                affected_systems=threat_detected.get("systems", []),
                recommendations=threat_detected.get("recommendations", []),
                confidence=threat_detected.get("confidence", 0.95)
            )
            
            result = ThreatAnalysisResult(
                analysis_id=analysis_id,
                status=AnalysisStatus.COMPLETED,
                sample=payload.sample,
                threat_detected=True,
                threat=threat,
                indicators=threat_detected.get("indicators", []),
                analysis_details=threat_detected.get("details", {}),
                processing_time_ms=150.5
            )
        else:
            result = ThreatAnalysisResult(
                analysis_id=analysis_id,
                status=AnalysisStatus.COMPLETED,
                sample=payload.sample,
                threat_detected=False,
                processing_time_ms=120.3
            )
        
        logger.info(f"Analysis completed: {analysis_id}")
        return result
    except Exception as e:
        logger.error(f"Error in threat analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get(
    "/api/v1/threats",
    response_model=List[ThreatDetection],
    tags=["Threat Detection"],
    summary="Retrieve detected threats"
)
async def get_threats(
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    threat_level: Optional[ThreatLevel] = Query(None),
    threat_type: Optional[ThreatType] = Query(None)
) -> List[ThreatDetection]:
    """
    Retrieve list of detected threats with optional filtering.
    
    Query Parameters:
    - limit: Maximum number of results (1-100)
    - offset: Offset for pagination
    - threat_level: Filter by severity level
    - threat_type: Filter by threat type
    """
    try:
        logger.info(f"Fetching threats - limit: {limit}, offset: {offset}")
        
        threats = _get_mock_threats(limit, offset, threat_level, threat_type)
        return threats
    except Exception as e:
        logger.error(f"Error retrieving threats: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve threats")


@app.get(
    "/api/v1/threats/{threat_id}",
    response_model=ThreatDetection,
    tags=["Threat Detection"],
    summary="Get threat details"
)
async def get_threat_details(threat_id: str) -> ThreatDetection:
    """Retrieve detailed information about a specific threat"""
    try:
        logger.info(f"Fetching threat details: {threat_id}")
        
        threat = _get_threat_by_id(threat_id)
        if not threat:
            raise HTTPException(status_code=404, detail="Threat not found")
        
        return threat
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving threat details: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve threat details")


# ==================== Threat Report Endpoints ====================

@app.post(
    "/api/v1/reports/generate",
    response_model=ThreatReport,
    tags=["Threat Reports"],
    summary="Generate threat report"
)
async def generate_threat_report(
    title: str = Query(..., description="Report title"),
    threat_ids: Optional[List[str]] = Query(None, description="Specific threat IDs to include")
) -> ThreatReport:
    """
    Generate comprehensive threat report.
    
    Includes:
    - Summary of detected threats
    - Threat statistics
    - Recommendations
    - Executive summary
    """
    try:
        logger.info(f"Generating threat report: {title}")
        
        threats = _get_mock_threats(limit=50)
        critical = sum(1 for t in threats if t.threat_level == ThreatLevel.CRITICAL)
        high = sum(1 for t in threats if t.threat_level == ThreatLevel.HIGH)
        
        report = ThreatReport(
            report_id=f"RPT-{datetime.utcnow().timestamp()}",
            title=title,
            threats=threats[:10],
            total_threats=len(threats),
            critical_count=critical,
            high_count=high,
            summary=f"Generated report with {len(threats)} detected threats",
            recommendations=[
                "Investigate critical threats immediately",
                "Isolate affected systems from network",
                "Apply security patches",
                "Review access logs"
            ]
        )
        
        logger.info(f"Report generated: {report.report_id}")
        return report
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        raise HTTPException(status_code=500, detail="Report generation failed")


@app.get(
    "/api/v1/reports/{report_id}",
    response_model=ThreatReport,
    tags=["Threat Reports"],
    summary="Retrieve threat report"
)
async def get_threat_report(report_id: str) -> ThreatReport:
    """Retrieve a previously generated threat report"""
    try:
        logger.info(f"Fetching report: {report_id}")
        
        report = _get_report_by_id(report_id)
        if not report:
            raise HTTPException(status_code=404, detail="Report not found")
        
        return report
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving report: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve report")


# ==================== Security Events Endpoints ====================

@app.post(
    "/api/v1/events/log",
    response_model=SecurityEvent,
    tags=["Security Events"],
    summary="Log security event"
)
async def log_security_event(
    event: SecurityEvent = Body(...)
) -> SecurityEvent:
    """
    Log a security event.
    
    Automatically processed for threat correlation and analysis.
    """
    try:
        logger.info(f"Logging security event: {event.event_id}")
        
        # Process event
        _process_security_event(event)
        
        return event
    except Exception as e:
        logger.error(f"Error logging security event: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to log event")


@app.get(
    "/api/v1/events",
    response_model=List[SecurityEvent],
    tags=["Security Events"],
    summary="Retrieve security events"
)
async def get_security_events(
    limit: int = Query(20, ge=1, le=100),
    event_type: Optional[str] = Query(None)
) -> List[SecurityEvent]:
    """Retrieve recent security events with optional filtering"""
    try:
        logger.info(f"Fetching security events - limit: {limit}")
        
        events = _get_mock_events(limit, event_type)
        return events
    except Exception as e:
        logger.error(f"Error retrieving events: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve events")


# ==================== Statistics Endpoints ====================

@app.get(
    "/api/v1/statistics/threats",
    tags=["Statistics"],
    summary="Get threat statistics"
)
async def get_threat_statistics():
    """Get threat detection statistics"""
    try:
        logger.info("Fetching threat statistics")
        
        stats = {
            "total_threats": 1247,
            "threats_today": 42,
            "threats_this_week": 185,
            "critical_threats": 3,
            "high_severity": 18,
            "medium_severity": 45,
            "low_severity": 89,
            "top_threat_types": {
                "MALWARE": 450,
                "PHISHING": 320,
                "SUSPICIOUS_ACTIVITY": 210,
                "DDoS": 155,
                "ANOMALY": 112
            },
            "avg_detection_time_ms": 145.3,
            "detection_rate_percentage": 96.5
        }
        
        return stats
    except Exception as e:
        logger.error(f"Error fetching statistics: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch statistics")


# ==================== Helper Functions ====================

def _analyze_sample(sample: str, sample_type: str) -> Optional[Dict[str, Any]]:
    """Mock threat analysis logic"""
    # Simulate threat detection based on sample patterns
    if any(keyword in sample.lower() for keyword in ["malware", "trojan", "virus"]):
        return {
            "type": ThreatType.MALWARE,
            "level": ThreatLevel.CRITICAL,
            "description": "Detected malicious code pattern",
            "confidence": 0.98,
            "indicators": [
                ThreatIndicator(
                    name="IOC_HASH",
                    value=sample,
                    indicator_type="file_hash",
                    confidence=0.98
                )
            ],
            "systems": ["SERVER-01", "WORKSTATION-05"],
            "recommendations": [
                "Isolate affected systems",
                "Scan with updated antivirus",
                "Review system logs"
            ],
            "details": {
                "file_type": "executable",
                "detected_families": ["Trojan.Generic"],
                "behavior_score": 95
            }
        }
    elif sample_type == "url" and any(x in sample for x in [".tk", ".ml", "phishing"]):
        return {
            "type": ThreatType.PHISHING,
            "level": ThreatLevel.HIGH,
            "description": "Suspected phishing URL",
            "confidence": 0.89,
            "indicators": [
                ThreatIndicator(
                    name="PHISHING_URL",
                    value=sample,
                    indicator_type="url",
                    confidence=0.89
                )
            ],
            "systems": [],
            "recommendations": [
                "Block URL in email gateway",
                "Warn users",
                "Report to phishing registry"
            ],
            "details": {
                "url_reputation": "malicious",
                "certificates": "invalid"
            }
        }
    else:
        return None


def _get_mock_threats(
    limit: int = 10,
    offset: int = 0,
    threat_level: Optional[ThreatLevel] = None,
    threat_type: Optional[ThreatType] = None
) -> List[ThreatDetection]:
    """Generate mock threat data"""
    threats = [
        ThreatDetection(
            threat_id="THR-1704844390",
            threat_type=ThreatType.MALWARE,
            threat_level=ThreatLevel.CRITICAL,
            detected_at=datetime.utcnow(),
            description="Critical malware detected in system memory",
            indicators=[
                ThreatIndicator(
                    name="MALWARE_HASH",
                    value="5d41402abc4b2a76b9719d911017c592",
                    indicator_type="hash",
                    confidence=0.98
                )
            ],
            affected_systems=["SERVER-01"],
            recommendations=["Isolate system", "Clean malware", "Monitor"],
            confidence=0.98
        ),
        ThreatDetection(
            threat_id="THR-1704844391",
            threat_type=ThreatType.PHISHING,
            threat_level=ThreatLevel.HIGH,
            detected_at=datetime.utcnow(),
            description="Phishing email detected",
            indicators=[
                ThreatIndicator(
                    name="PHISHING_URL",
                    value="http://malicious-domain.tk",
                    indicator_type="url",
                    confidence=0.85
                )
            ],
            recommendations=["Block sender", "Warn users"],
            confidence=0.85
        )
    ]
    
    return threats[offset:offset + limit]


def _get_threat_by_id(threat_id: str) -> Optional[ThreatDetection]:
    """Retrieve threat by ID"""
    threats = _get_mock_threats(limit=100)
    for threat in threats:
        if threat.threat_id == threat_id:
            return threat
    return None


def _get_report_by_id(report_id: str) -> Optional[ThreatReport]:
    """Retrieve report by ID"""
    threats = _get_mock_threats(limit=10)
    return ThreatReport(
        report_id=report_id,
        title="Security Threat Report",
        threats=threats,
        total_threats=len(threats),
        critical_count=sum(1 for t in threats if t.threat_level == ThreatLevel.CRITICAL),
        high_count=sum(1 for t in threats if t.threat_level == ThreatLevel.HIGH),
        summary="Comprehensive threat analysis report",
        recommendations=["Review logs", "Update defenses"]
    )


def _get_mock_events(
    limit: int = 20,
    event_type: Optional[str] = None
) -> List[SecurityEvent]:
    """Generate mock security events"""
    events = [
        SecurityEvent(
            event_id="EVT-001",
            event_type="SUSPICIOUS_LOGIN",
            severity=ThreatLevel.HIGH,
            source="192.168.1.100",
            description="Multiple failed login attempts detected",
            metadata={"attempts": 5, "user": "admin"}
        ),
        SecurityEvent(
            event_id="EVT-002",
            event_type="NETWORK_ANOMALY",
            severity=ThreatLevel.MEDIUM,
            source="192.168.1.50",
            description="Unusual network traffic detected",
            metadata={"bandwidth": "2.5GB", "duration": "15m"}
        )
    ]
    
    return events[:limit]


def _process_security_event(event: SecurityEvent) -> None:
    """Process security event for threat correlation"""
    logger.info(f"Processing security event: {event.event_id}")
    # Implement event correlation logic here


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
