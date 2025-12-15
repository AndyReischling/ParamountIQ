import { SPSPatientCase } from './spsPatientGenerator';
import OpenAI from 'openai';

interface Message {
  role: 'patient' | 'provider';
  content: string;
  timestamp: Date;
}

export interface SPSScore {
  rubrics: {
    historyGathering: RubricScore;
    clinicalReasoning: RubricScore;
    diagnosisAssessment: RubricScore;
    communicationEmpathy: RubricScore;
    safetyRedFlags: RubricScore;
    counselingNextSteps: RubricScore;
    timeManagement: RubricScore;
  };
  totalScore: number;
  caseId: string;
  patientSummary: string;
  strengths: string[];
  improvements: string[];
  missedOpportunities: string[];
  groundTruthDiagnosis: string;
  acceptedDifferentials: string[];
  quotedEvidence: QuotedEvidence[];
  feedbackNote: string;
  suggestedFocus: string;
}

interface RubricScore {
  score: number; // 0-5
  maxScore: number; // always 5
  weight: number; // percentage weight
  feedback: string;
  evidence: string[];
}

interface QuotedEvidence {
  quote: string;
  rubric: string;
  explanation: string;
}

const SCORING_WEIGHTS = {
  historyGathering: 0.20,
  clinicalReasoning: 0.20,
  diagnosisAssessment: 0.20,
  communicationEmpathy: 0.15,
  safetyRedFlags: 0.15,
  counselingNextSteps: 0.05,
  timeManagement: 0.05
};

import { getStoredApiKey } from './apiKeyManager';

// Get API key from localStorage
const getApiKey = (): string => {
  const storedKey = getStoredApiKey();
  if (storedKey && storedKey !== 'DEMO_MODE') {
    return storedKey;
  }
  
  if (typeof import.meta !== 'undefined' && import.meta.env?.VITE_OPENAI_API_KEY) {
    return import.meta.env.VITE_OPENAI_API_KEY;
  }
  
  return 'DEMO_MODE';
};

const createOpenAIClient = () => {
  const apiKey = getApiKey();
  if (apiKey === 'DEMO_MODE') {
    return null;
  }
  return new OpenAI({
    apiKey,
    dangerouslyAllowBrowser: true
  });
};

export async function scoreSPSCase(
  patientCase: SPSPatientCase,
  conversationHistory: Message[]
): Promise<SPSScore> {
  const openai = createOpenAIClient();
  
  // If in demo mode, use fallback scoring
  if (!openai) {
    console.log('Running in demo mode - using fallback scoring');
    return generateFallbackScores(patientCase, conversationHistory);
  }
  
  try {
    // Build the full conversation transcript
    const transcript = conversationHistory
      .map(msg => `${msg.role === 'provider' ? 'Student' : 'Patient'}: ${msg.content}`)
      .join('\n\n');
    
    const scoringPrompt = buildScoringPrompt(patientCase, transcript);
    
    const completion = await openai.chat.completions.create({
      model: 'gpt-4o',
      messages: [
        {
          role: 'system',
          content: 'You are an expert clinical educator evaluating a medical student\'s patient encounter. Provide detailed, constructive feedback with specific evidence from the conversation.'
        },
        {
          role: 'user',
          content: scoringPrompt
        }
      ],
      temperature: 0.3,
      max_tokens: 2500,
    });
    
    const aiResponse = completion.choices[0].message.content || '';
    
    // Parse AI response and structure the scores
    const scores = parseAIScoring(aiResponse, patientCase, conversationHistory);
    
    return scores;
    
  } catch (error) {
    console.error('Scoring error:', error);
    return generateFallbackScores(patientCase, conversationHistory);
  }
}

function buildScoringPrompt(patientCase: SPSPatientCase, transcript: string): string {
  return `You are evaluating a medical student's clinical encounter. The student interviewed a standardized patient with the following actual diagnosis: ${patientCase.diagnosis}.

CASE GROUND TRUTH:
- Diagnosis: ${patientCase.diagnosis}
- Accepted Differentials: ${patientCase.acceptedDifferentials.join(', ')}
- Critical Red Flags: ${patientCase.redFlags.join(', ')}
- Key History Elements: OLDCARTS components, medications, allergies, PMH, PSH, family history, social history

CONVERSATION TRANSCRIPT:
${transcript}

SCORING RUBRICS (Each scored 0-5):

1. HISTORY GATHERING (Weight: 20%)
   - Obtained comprehensive OLDCARTS
   - Asked about medications, allergies
   - Explored PMH, PSH, family history, social history
   - Asked relevant systems review questions
   Score: [0-5]
   Evidence: [specific quotes]
   Feedback: [detailed explanation]

2. CLINICAL REASONING (Weight: 20%)
   - Demonstrated logical thinking
   - Asked questions building toward diagnosis
   - Recognized patterns and connections
   - Appropriate breadth of differential
   Score: [0-5]
   Evidence: [specific quotes]
   Feedback: [detailed explanation]

3. DIAGNOSIS/ASSESSMENT (Weight: 20%)
   - Arrived at correct diagnosis or reasonable differential
   - Articulated clinical reasoning
   - Discussed next diagnostic steps appropriately
   Score: [0-5]
   Evidence: [specific quotes]
   Feedback: [detailed explanation]

4. COMMUNICATION & EMPATHY (Weight: 15%)
   - Used empathetic language
   - Active listening demonstrated
   - Appropriate rapport building
   - Clear, jargon-free explanations
   - Addressed patient concerns
   Score: [0-5]
   Evidence: [specific quotes]
   Feedback: [detailed explanation]

5. SAFETY & RED FLAGS (Weight: 15%)
   - Identified critical red flags: ${patientCase.redFlags.join(', ')}
   - Assessed severity appropriately
   - Addressed urgent concerns
   Score: [0-5]
   Evidence: [specific quotes]
   Feedback: [detailed explanation]

6. COUNSELING & NEXT STEPS (Weight: 5%)
   - Explained plan clearly
   - Discussed follow-up
   - Provided appropriate counseling
   Score: [0-5]
   Evidence: [specific quotes]
   Feedback: [detailed explanation]

7. TIME MANAGEMENT (Weight: 5%)
   - Efficient questioning
   - Avoided excessive tangents
   - Covered essential elements
   Score: [0-5]
   Evidence: [specific quotes]
   Feedback: [detailed explanation]

Provide your evaluation in this exact format:

RUBRIC SCORES:
HistoryGathering: [0-5] | Evidence: [quotes] | Feedback: [explanation]
ClinicalReasoning: [0-5] | Evidence: [quotes] | Feedback: [explanation]
DiagnosisAssessment: [0-5] | Evidence: [quotes] | Feedback: [explanation]
CommunicationEmpathy: [0-5] | Evidence: [quotes] | Feedback: [explanation]
SafetyRedFlags: [0-5] | Evidence: [quotes] | Feedback: [explanation]
CounselingNextSteps: [0-5] | Evidence: [quotes] | Feedback: [explanation]
TimeManagement: [0-5] | Evidence: [quotes] | Feedback: [explanation]

STRENGTHS:
- [bullet point]
- [bullet point]

IMPROVEMENTS:
- [bullet point]
- [bullet point]

MISSED OPPORTUNITIES:
- [bullet point]
- [bullet point]

FEEDBACK NOTE (≤250 words):
[Concise constructive feedback]

SUGGESTED FOCUS:
[What to practice next]`;
}

function parseAIScoring(aiResponse: string, patientCase: SPSPatientCase, history: Message[]): SPSScore {
  // Parse the AI response and extract structured scores
  // This is a simplified parser - in production you'd use more robust parsing
  
  const rubrics = {
    historyGathering: extractRubricScore(aiResponse, 'HistoryGathering', SCORING_WEIGHTS.historyGathering),
    clinicalReasoning: extractRubricScore(aiResponse, 'ClinicalReasoning', SCORING_WEIGHTS.clinicalReasoning),
    diagnosisAssessment: extractRubricScore(aiResponse, 'DiagnosisAssessment', SCORING_WEIGHTS.diagnosisAssessment),
    communicationEmpathy: extractRubricScore(aiResponse, 'CommunicationEmpathy', SCORING_WEIGHTS.communicationEmpathy),
    safetyRedFlags: extractRubricScore(aiResponse, 'SafetyRedFlags', SCORING_WEIGHTS.safetyRedFlags),
    counselingNextSteps: extractRubricScore(aiResponse, 'CounselingNextSteps', SCORING_WEIGHTS.counselingNextSteps),
    timeManagement: extractRubricScore(aiResponse, 'TimeManagement', SCORING_WEIGHTS.timeManagement)
  };
  
  // Calculate total score out of 100
  const totalScore = Object.values(rubrics).reduce((sum, rubric) => {
    return sum + (rubric.score / rubric.maxScore) * rubric.weight * 100;
  }, 0);
  
  const strengths = extractBulletPoints(aiResponse, 'STRENGTHS:');
  const improvements = extractBulletPoints(aiResponse, 'IMPROVEMENTS:');
  const missedOpportunities = extractBulletPoints(aiResponse, 'MISSED OPPORTUNITIES:');
  
  const feedbackNote = extractSection(aiResponse, 'FEEDBACK NOTE', 'SUGGESTED FOCUS:');
  const suggestedFocus = extractSection(aiResponse, 'SUGGESTED FOCUS:', '---');
  
  const quotedEvidence = extractQuotedEvidence(aiResponse);
  
  return {
    rubrics,
    totalScore: Math.round(totalScore),
    caseId: patientCase.caseId,
    patientSummary: `${patientCase.name}, ${patientCase.age}yo ${patientCase.gender}, presenting with ${patientCase.chiefComplaint}`,
    strengths,
    improvements,
    missedOpportunities,
    groundTruthDiagnosis: patientCase.diagnosis,
    acceptedDifferentials: patientCase.acceptedDifferentials,
    quotedEvidence,
    feedbackNote: feedbackNote || 'Good effort on this case.',
    suggestedFocus: suggestedFocus || 'Continue practicing systematic history taking.'
  };
}

function extractRubricScore(text: string, rubricName: string, weight: number): RubricScore {
  const regex = new RegExp(`${rubricName}:\\s*(\\d)\\s*\\|\\s*Evidence:\\s*([^|]+)\\|\\s*Feedback:\\s*([^\\n]+)`, 'i');
  const match = text.match(regex);
  
  if (match) {
    return {
      score: parseInt(match[1]) || 3,
      maxScore: 5,
      weight: weight * 100,
      feedback: match[3].trim(),
      evidence: [match[2].trim()]
    };
  }
  
  return {
    score: 3,
    maxScore: 5,
    weight: weight * 100,
    feedback: 'Assessment in progress',
    evidence: []
  };
}

function extractBulletPoints(text: string, sectionHeader: string): string[] {
  const sectionMatch = text.indexOf(sectionHeader);
  if (sectionMatch === -1) return [];
  
  const afterHeader = text.substring(sectionMatch + sectionHeader.length);
  const nextSection = afterHeader.search(/\n[A-Z]+:/);
  const sectionText = nextSection !== -1 ? afterHeader.substring(0, nextSection) : afterHeader.substring(0, 500);
  
  const bullets = sectionText
    .split('\n')
    .filter(line => line.trim().startsWith('-'))
    .map(line => line.trim().substring(1).trim());
  
  return bullets.length > 0 ? bullets : ['Continue developing skills in this area'];
}

function extractSection(text: string, startMarker: string, endMarker: string): string {
  const startIndex = text.indexOf(startMarker);
  if (startIndex === -1) return '';
  
  const contentStart = startIndex + startMarker.length;
  const endIndex = text.indexOf(endMarker, contentStart);
  
  const content = endIndex !== -1 
    ? text.substring(contentStart, endIndex)
    : text.substring(contentStart, contentStart + 500);
  
  return content.trim();
}

function extractQuotedEvidence(text: string): QuotedEvidence[] {
  // Extract evidence quotes from the response
  const evidence: QuotedEvidence[] = [];
  
  const rubricNames = [
    'HistoryGathering',
    'ClinicalReasoning',
    'DiagnosisAssessment',
    'CommunicationEmpathy',
    'SafetyRedFlags',
    'CounselingNextSteps',
    'TimeManagement'
  ];
  
  rubricNames.forEach(rubric => {
    const regex = new RegExp(`${rubric}:.*?Evidence:\\s*([^|]+)\\|\\s*Feedback:\\s*([^\\n]+)`, 'i');
    const match = text.match(regex);
    
    if (match && match[1]) {
      const quotes = match[1].split(';').map(q => q.trim()).filter(q => q.length > 0);
      quotes.forEach(quote => {
        evidence.push({
          quote,
          rubric,
          explanation: match[2].trim()
        });
      });
    }
  });
  
  return evidence;
}

function generateFallbackScores(patientCase: SPSPatientCase, history: Message[]): SPSScore {
  // Fallback scoring if AI fails
  const providerMessages = history.filter(m => m.role === 'provider');
  const baseScore = Math.min(5, Math.floor(providerMessages.length / 2) + 1);
  
  const createRubric = (weight: number): RubricScore => ({
    score: baseScore,
    maxScore: 5,
    weight: weight * 100,
    feedback: 'Good attempt. Continue practicing systematic history taking.',
    evidence: []
  });
  
  return {
    rubrics: {
      historyGathering: createRubric(SCORING_WEIGHTS.historyGathering),
      clinicalReasoning: createRubric(SCORING_WEIGHTS.clinicalReasoning),
      diagnosisAssessment: createRubric(SCORING_WEIGHTS.diagnosisAssessment),
      communicationEmpathy: createRubric(SCORING_WEIGHTS.communicationEmpathy),
      safetyRedFlags: createRubric(SCORING_WEIGHTS.safetyRedFlags),
      counselingNextSteps: createRubric(SCORING_WEIGHTS.counselingNextSteps),
      timeManagement: createRubric(SCORING_WEIGHTS.timeManagement)
    },
    totalScore: (baseScore / 5) * 100,
    caseId: patientCase.caseId,
    patientSummary: `${patientCase.name}, ${patientCase.age}yo ${patientCase.gender}, presenting with ${patientCase.chiefComplaint}`,
    strengths: ['Engaged with the patient', 'Asked relevant questions'],
    improvements: ['Continue developing systematic approach', 'Practice OLDCARTS framework'],
    missedOpportunities: ['Could explore red flags more thoroughly'],
    groundTruthDiagnosis: patientCase.diagnosis,
    acceptedDifferentials: patientCase.acceptedDifferentials,
    quotedEvidence: [],
    feedbackNote: 'You demonstrated engagement with the patient. Continue practicing structured history taking and clinical reasoning.',
    suggestedFocus: 'Practice OLDCARTS framework and recognizing red flags'
  };
}
