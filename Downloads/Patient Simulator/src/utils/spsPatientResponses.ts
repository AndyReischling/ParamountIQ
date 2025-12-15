import { SPSPatientCase } from './spsPatientGenerator';
import OpenAI from 'openai';

interface Message {
  role: 'patient' | 'provider';
  content: string;
  timestamp: Date;
}

interface PatientResponse {
  content: string;
  shouldEnd: boolean;
}

import { getStoredApiKey } from './apiKeyManager';

// Get API key from localStorage
const getApiKey = (): string => {
  const storedKey = getStoredApiKey();
  if (storedKey && storedKey !== 'DEMO_MODE') {
    return storedKey;
  }
  
  // Check environment variables as fallback
  if (typeof import.meta !== 'undefined' && import.meta.env?.VITE_OPENAI_API_KEY) {
    return import.meta.env.VITE_OPENAI_API_KEY;
  }
  
  return 'DEMO_MODE';
};

// Initialize OpenAI client (will be reinitiated on each call with fresh key)
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

export async function getSPSPatientResponse(
  patientCase: SPSPatientCase,
  providerMessage: string,
  conversationHistory: Message[]
): Promise<PatientResponse> {
  const openai = createOpenAIClient();
  
  // If in demo mode or no API key, use fallback
  if (!openai) {
    console.log('Running in demo mode - using fallback responses');
    return {
      content: generateFallbackResponse(patientCase),
      shouldEnd: checkIfShouldEnd(providerMessage, conversationHistory.length)
    };
  }
  
  try {
    const systemPrompt = buildSPSSystemPrompt(patientCase);
    const conversationContext = buildConversationContext(conversationHistory);
    
    const completion = await openai.chat.completions.create({
      model: 'gpt-4o-mini',
      messages: [
        { role: 'system', content: systemPrompt },
        ...conversationContext,
        { role: 'user', content: providerMessage }
      ],
      temperature: 0.8,
      max_tokens: 250,
      presence_penalty: 0.6,
      frequency_penalty: 0.3,
    });
    
    const response = completion.choices[0].message.content || generateFallbackResponse(patientCase);
    
    // Check if conversation should end
    const shouldEnd = checkIfShouldEnd(providerMessage, conversationHistory.length);
    
    return {
      content: response,
      shouldEnd
    };
    
  } catch (error) {
    console.error('OpenAI API error:', error);
    return {
      content: generateFallbackResponse(patientCase),
      shouldEnd: false
    };
  }
}

function buildSPSSystemPrompt(patientCase: SPSPatientCase): string {
  return `You are a Standardized Patient in a clinical education simulation. You are roleplaying as a real patient in an ${patientCase.setting === 'emergency_department' ? 'emergency department' : 'outpatient clinic'} visit.

PATIENT IDENTITY:
- Name: ${patientCase.name}
- Age: ${patientCase.age}
- Gender: ${patientCase.gender} (${patientCase.pronouns})
- Ethnicity: ${patientCase.ethnicity}
- Occupation: ${patientCase.occupation}
- Language Level: ${patientCase.languageLevel}

CHIEF COMPLAINT:
${patientCase.chiefComplaint}

CLINICAL DETAILS (DO NOT VOLUNTEER - ONLY SHARE IF SPECIFICALLY ASKED):

OLDCARTS:
- Onset: ${patientCase.onset}
- Location: ${patientCase.location}
- Duration: ${patientCase.duration}
- Character: ${patientCase.character}
- Aggravating Factors: ${patientCase.aggravatingFactors.join(', ')}
- Relieving Factors: ${patientCase.relievingFactors.join(', ')}
- Timing: ${patientCase.timing}
- Severity: ${patientCase.severity}

CURRENT VITALS (only share if asked):
- BP: ${patientCase.vitals.bloodPressure}
- HR: ${patientCase.vitals.heartRate}
- RR: ${patientCase.vitals.respiratoryRate}
- Temp: ${patientCase.vitals.temperature}°C
- O2 Sat: ${patientCase.vitals.oxygenSaturation}%
${patientCase.vitals.painScale ? `- Pain: ${patientCase.vitals.painScale}/10` : ''}

MEDICATIONS: ${patientCase.medications.join(', ')}
ALLERGIES: ${patientCase.allergies.join(', ')}

PAST MEDICAL HISTORY: ${patientCase.pastMedicalHistory.join(', ')}
PAST SURGICAL HISTORY: ${patientCase.pastSurgicalHistory.join(', ')}

FAMILY HISTORY: ${patientCase.familyHistory}

SOCIAL HISTORY:
- Smoking: ${patientCase.socialHistory.smoking}
- Alcohol: ${patientCase.socialHistory.alcohol}
- Drugs: ${patientCase.socialHistory.drugs}
- Living Situation: ${patientCase.socialHistory.living}

${patientCase.sexualHistory ? `SEXUAL HISTORY: ${patientCase.sexualHistory}` : ''}

PHYSICAL EXAM FINDINGS (only share if doctor examines that area):
${Object.entries(patientCase.physicalExamFindings).map(([area, finding]) => `- ${area}: ${finding}`).join('\n')}

PERSONALITY & COMMUNICATION:
- Emotional State: ${patientCase.emotionalState}
- Personality: ${patientCase.personalityTraits.join(', ')}
- Language Level: ${getLanguageGuidance(patientCase.languageLevel)}

CRITICAL RULES FOR YOUR RESPONSES:

1. **Answer naturally as a real patient would** - Use lay language, not medical terminology
2. **Only share information when specifically asked** - Don't volunteer details
3. **Stay in character** - React emotionally consistent with your personality
4. **Be realistic about knowledge** - You don't know medical terms or what tests mean
5. **Physical exam responses** - If asked to examine you, describe what they would find
6. **Don't offer diagnosis** - You're the patient, not the doctor
7. **Vary your responses** - Don't repeat the same phrases
8. **Show emotion** - Express fear, worry, relief, frustration as appropriate
9. **Ask questions** - Real patients ask about their condition
10. **Be human** - Include hesitations, uncertainties, personal concerns

DISTRACTORS (mention if relevant to conversation):
${patientCase.distractors.join(', ')}

RED FLAGS (present but don't emphasize unless asked):
${patientCase.redFlags.join(', ')}

Remember: You are ${patientCase.name}, a ${patientCase.age}-year-old ${patientCase.occupation} who came in because of ${patientCase.chiefComplaint}. Respond only as this patient would, using simple language and showing appropriate emotion.`;
}

function getLanguageGuidance(level: SPSPatientCase['languageLevel']): string {
  const guidance = {
    'fluent': 'Speak clearly with good vocabulary',
    'moderate': 'Use simpler words, occasional grammar mistakes',
    'limited': 'Struggle with complex medical terms, use basic English',
    'interpreter_needed': 'Very limited English, may need things repeated'
  };
  return guidance[level];
}

function buildConversationContext(history: Message[]): Array<{role: 'user' | 'assistant', content: string}> {
  const recentHistory = history.slice(-8);
  
  return recentHistory.map(msg => ({
    role: msg.role === 'provider' ? 'user' as const : 'assistant' as const,
    content: msg.content
  }));
}

function generateFallbackResponse(patientCase: SPSPatientCase): string {
  const responses: Record<string, string> = {
    'anxious': "I'm sorry, I'm just really worried. Could you say that again?",
    'calm': "I'm not sure I understand. Could you explain?",
    'distressed': "This is all so overwhelming. What does that mean?",
    'worried': "I'm concerned about what this could be. Can you help me understand?",
    'frustrated': "I just want to know what's wrong. Can you tell me?",
    'scared': "I'm frightened. What should I do?",
    'embarrassed': "I'm sorry, could you repeat that?"
  };
  
  return responses[patientCase.emotionalState] || "Could you explain that differently?";
}

function checkIfShouldEnd(message: string, turnCount: number): boolean {
  const endPhrases = /\b(goodbye|take care|see you|follow.?up|that'?s all|appointment scheduled|prescription|we'?re done)\b/i;
  return endPhrases.test(message) || turnCount >= 20;
}
