import { PatientProfile } from './patientGenerator';

interface Message {
  role: 'patient' | 'provider';
  content: string;
  timestamp: Date;
}

export interface ScoringDetail {
  element: string;
  maxPoints: number;
  earnedPoints: number;
  explanation: string;
  found: boolean;
}

export interface ConversationAnalysis {
  totalScore: number;
  scores: {
    start: number;
    heart: number;
    care: number;
    wow: number;
  };
  scoringDetails: {
    start: ScoringDetail[];
    heart: ScoringDetail[];
    care: ScoringDetail[];
    wow: ScoringDetail[];
  };
  feedback: {
    strengths: string[];
    improvements: string[];
    missed: string[];
    overall: string;
  };
}

export function analyzeConversation(
  messages: Message[],
  patient: PatientProfile
): ConversationAnalysis {
  const providerMessages = messages.filter(m => m.role === 'provider');
  
  if (providerMessages.length === 0) {
    return {
      totalScore: 0,
      scores: { start: 0, heart: 0, care: 0, wow: 0 },
      feedback: {
        strengths: [],
        improvements: ['No provider messages to analyze'],
        missed: [],
        overall: 'Unable to assess communication skills without provider responses.'
      }
    };
  }

  const allProviderText = providerMessages.map(m => m.content).join(' ').toLowerCase();
  const firstMessage = providerMessages[0]?.content.toLowerCase() || '';
  const lastMessage = providerMessages[providerMessages.length - 1]?.content.toLowerCase() || '';

  // S.T.A.R.T. Analysis (25 points)
  let startScore = 0;
  const startFeedback = analyzeSTART(firstMessage, allProviderText, patient);
  startScore = startFeedback.score;

  // H.E.A.R.T. Analysis (25 points)
  let heartScore = 0;
  const heartFeedback = analyzeHEART(allProviderText, providerMessages, patient);
  heartScore = heartFeedback.score;

  // C.A.R.E. Analysis (25 points)
  let careScore = 0;
  const careFeedback = analyzeCARE(allProviderText, providerMessages);
  careScore = careFeedback.score;

  // WOW Analysis (25 points)
  let wowScore = 0;
  const wowFeedback = analyzeWOW(allProviderText, lastMessage, providerMessages);
  wowScore = wowFeedback.score;

  const totalScore = Math.round(startScore + heartScore + careScore + wowScore);

  // Compile all feedback
  const strengths: string[] = [];
  const improvements: string[] = [];
  const missed: string[] = [];

  [startFeedback, heartFeedback, careFeedback, wowFeedback].forEach(fb => {
    strengths.push(...fb.strengths);
    improvements.push(...fb.improvements);
    missed.push(...fb.missed);
  });

  const overall = generateOverallAssessment(totalScore, patient);

  return {
    totalScore,
    scores: {
      start: Math.round(startScore),
      heart: Math.round(heartScore),
      care: Math.round(careScore),
      wow: Math.round(wowScore)
    },
    scoringDetails: {
      start: startFeedback.details,
      heart: heartFeedback.details,
      care: careFeedback.details,
      wow: wowFeedback.details
    },
    feedback: {
      strengths,
      improvements,
      missed,
      overall
    }
  };
}

function analyzeSTART(firstMessage: string, allText: string, patient: PatientProfile) {
  let score = 0;
  const strengths: string[] = [];
  const improvements: string[] = [];
  const missed: string[] = [];
  const details: ScoringDetail[] = [];

  // Smile/Greet warmly (5 points)
  const hasGreeting = /\b(hi|hello|good morning|good afternoon|good evening)\b/i.test(firstMessage);
  details.push({
    element: 'Smile & Greet warmly',
    maxPoints: 5,
    earnedPoints: hasGreeting ? 5 : 0,
    explanation: hasGreeting 
      ? 'Used a warm greeting in opening statement' 
      : 'No greeting found. Start with "Hello" or "Good morning" to establish warmth',
    found: hasGreeting
  });
  if (hasGreeting) {
    score += 5;
    strengths.push('Greeted the patient warmly');
  } else {
    improvements.push('Start with a warm greeting');
    missed.push('Opening greeting (S.T.A.R.T.: Smile and greet warmly)');
  }

  // Tell name, role, and what to expect (5 points)
  const hasIntroduction = /\b(my name is|i'm|i am)\b/i.test(firstMessage);
  const hasRole = /\b(doctor|nurse|provider|physician|practitioner)\b/i.test(allText);
  const hasExpectation = /\b(here to help|here to discuss|today we'll|we're going to)\b/i.test(allText);
  
  const introPoints = (hasIntroduction ? 3 : 0) + ((hasRole || hasExpectation) ? 2 : 0);
  details.push({
    element: 'Tell name, role, and what to expect',
    maxPoints: 5,
    earnedPoints: introPoints,
    explanation: `${hasIntroduction ? '✓ Introduced name (3 pts)' : '✗ Missing name introduction (0/3 pts)'} | ${(hasRole || hasExpectation) ? '✓ Explained role/expectations (2 pts)' : '✗ Missing role clarification (0/2 pts)'}`,
    found: hasIntroduction && (hasRole || hasExpectation)
  });
  
  if (hasIntroduction) {
    score += 3;
    strengths.push('Introduced yourself to the patient');
  } else {
    improvements.push('Introduce yourself by name and role');
  }
  
  if (hasRole || hasExpectation) {
    score += 2;
  } else {
    missed.push('Explaining your role or what to expect from the visit');
  }

  // Active listening (5 points)
  const hasActiveListening = /\b(tell me more|go on|i hear you|i'm listening|help me understand)\b/i.test(allText);
  const hasOpenQuestions = /\b(how|what|when|tell me|describe|explain)\b.*\?/i.test(allText);
  
  const listeningPoints = (hasActiveListening ? 3 : 0) + (hasOpenQuestions ? 2 : 0);
  details.push({
    element: 'Active listening',
    maxPoints: 5,
    earnedPoints: listeningPoints,
    explanation: `${hasActiveListening ? '✓ Used active listening phrases (3 pts)' : '✗ Missing phrases like "tell me more" (0/3 pts)'} | ${hasOpenQuestions ? '✓ Asked open-ended questions (2 pts)' : '✗ No open-ended questions found (0/2 pts)'}`,
    found: hasActiveListening || hasOpenQuestions
  });
  
  if (hasActiveListening) {
    score += 3;
    strengths.push('Demonstrated active listening');
  }
  
  if (hasOpenQuestions) {
    score += 2;
    strengths.push('Asked open-ended questions');
  } else {
    improvements.push('Use open-ended questions to encourage the patient to share');
  }

  // Rapport building (5 points)
  const hasEmpathy = /\b(understand|hear you|sounds|must be|difficult|challenging|frustrating)\b/i.test(allText);
  const usesPatientName = allText.includes(patient.name.toLowerCase().split(' ')[0].toLowerCase());
  
  const rapportPoints = (hasEmpathy ? 3 : 0) + (usesPatientName ? 2 : 0);
  details.push({
    element: 'Rapport building',
    maxPoints: 5,
    earnedPoints: rapportPoints,
    explanation: `${hasEmpathy ? '✓ Expressed empathy (3 pts)' : '✗ No empathetic language detected (0/3 pts)'} | ${usesPatientName ? '✓ Used patient\'s name (2 pts)' : '✗ Did not use patient\'s name (0/2 pts)'}`,
    found: hasEmpathy || usesPatientName
  });
  
  if (hasEmpathy) {
    score += 3;
    strengths.push('Showed empathy and understanding');
  } else {
    improvements.push('Express empathy for the patient\'s situation');
  }
  
  if (usesPatientName) {
    score += 2;
    strengths.push('Used the patient\'s name personally');
  } else {
    missed.push('Using the patient\'s name to build rapport');
  }

  // Thank you (5 points)
  const hasThanks = /\b(thank you|thanks|appreciate)\b/i.test(allText);
  details.push({
    element: 'Thank the patient',
    maxPoints: 5,
    earnedPoints: hasThanks ? 5 : 0,
    explanation: hasThanks 
      ? 'Expressed gratitude or appreciation to the patient' 
      : 'No "thank you" or appreciation expressed. Patients value being thanked for their time',
    found: hasThanks
  });
  
  if (hasThanks) {
    score += 5;
    strengths.push('Expressed gratitude to the patient');
  } else {
    improvements.push('Thank the patient for sharing their concerns');
  }

  return { score, strengths, improvements, missed, details };
}

function analyzeHEART(allText: string, messages: Message[], patient: PatientProfile) {
  let score = 0;
  const strengths: string[] = [];
  const improvements: string[] = [];
  const missed: string[] = [];
  const details: ScoringDetail[] = [];

  // Hear (5 points)
  const hasReflection = /\b(so what i'm hearing|it sounds like|if i understand|let me make sure)\b/i.test(allText);
  const hasClarifying = /\b(can you clarify|help me understand|tell me more about)\b/i.test(allText);
  
  const hearPoints = (hasReflection ? 3 : 0) + (hasClarifying ? 2 : 0);
  details.push({
    element: 'H - Hear the story',
    maxPoints: 5,
    earnedPoints: hearPoints,
    explanation: `${hasReflection ? '✓ Reflected understanding (3 pts)' : '✗ No reflective statements (0/3 pts)'} | ${hasClarifying ? '✓ Asked clarifying questions (2 pts)' : '✗ No clarifying questions (0/2 pts)'}`,
    found: hasReflection || hasClarifying
  });
  
  if (hasReflection) {
    score += 3;
    strengths.push('Reflected back patient\'s concerns');
  } else {
    improvements.push('Reflect back what you\'re hearing to confirm understanding');
  }
  
  if (hasClarifying) {
    score += 2;
  } else {
    missed.push('Asking clarifying questions to fully understand concerns');
  }

  // Empathize (5 points)
  const empathyPhrases = /\b(i understand|i can imagine|that must be|sounds difficult|i hear that|that's understandable)\b/i.test(allText);
  const emotionNaming = new RegExp(`\\b(${patient.emotionalState.toLowerCase()}|frustrated|worried|concerned|anxious)\\b`, 'i').test(allText);
  
  const empathyPoints = (empathyPhrases ? 3 : 0) + (emotionNaming ? 2 : 0);
  details.push({
    element: 'E - Empathize',
    maxPoints: 5,
    earnedPoints: empathyPoints,
    explanation: `${empathyPhrases ? '✓ Used empathetic phrases (3 pts)' : '✗ Missing empathy language like "I understand" (0/3 pts)'} | ${emotionNaming ? '✓ Named emotions (2 pts)' : '✗ Did not acknowledge patient emotions (0/2 pts)'}`,
    found: empathyPhrases || emotionNaming
  });
  
  if (empathyPhrases) {
    score += 3;
    strengths.push('Used empathetic language');
  } else {
    improvements.push('Use phrases that show empathy (e.g., "I understand," "That must be difficult")');
  }
  
  if (emotionNaming) {
    score += 2;
    strengths.push('Named or acknowledged the patient\'s emotions (S.A.V.E. technique)');
  }

  // Apologize (5 points)
  const hasApology = /\b(i'm sorry|apologize|regret|unfortunate)\b/i.test(allText);
  const hasOwnership = /\b(we should have|i should have|that wasn't|we could have done better)\b/i.test(allText);
  
  const apologizePoints = (hasApology || hasOwnership) ? 5 : 0;
  const needsApology = patient.emotionalState === 'Angry' || patient.emotionalState === 'Frustrated';
  details.push({
    element: 'A - Apologize',
    maxPoints: 5,
    earnedPoints: apologizePoints,
    explanation: (hasApology || hasOwnership) 
      ? 'Offered appropriate apology or acknowledgment of difficult experience' 
      : needsApology 
        ? 'Patient is frustrated/angry - an apology would validate their experience'
        : 'No apology needed for this patient\'s emotional state, but always appropriate',
    found: hasApology || hasOwnership
  });
  
  if (hasApology || hasOwnership) {
    score += 5;
    strengths.push('Apologized appropriately for patient\'s experience');
  } else if (needsApology) {
    improvements.push('Apologize for the patient\'s difficult experience when appropriate');
  }

  // Respond (5 points)
  const hasActionPlan = /\b(we will|i will|next steps|plan|let's|going to|schedule|prescribe|refer)\b/i.test(allText);
  const hasExplanation = /\b(because|the reason|this means|what this means is|here's why)\b/i.test(allText);
  
  const respondPoints = (hasActionPlan ? 3 : 0) + (hasExplanation ? 2 : 0);
  details.push({
    element: 'R - Respond with a plan',
    maxPoints: 5,
    earnedPoints: respondPoints,
    explanation: `${hasActionPlan ? '✓ Outlined action plan/next steps (3 pts)' : '✗ No clear action plan provided (0/3 pts)'} | ${hasExplanation ? '✓ Explained reasoning (2 pts)' : '✗ Did not explain "why" behind recommendations (0/2 pts)'}`,
    found: hasActionPlan || hasExplanation
  });
  
  if (hasActionPlan) {
    score += 3;
    strengths.push('Provided clear action plan and next steps');
  } else {
    improvements.push('Clearly outline next steps and action items');
  }
  
  if (hasExplanation) {
    score += 2;
    strengths.push('Explained reasoning behind recommendations');
  }

  // Thank (5 points)
  const lastMessages = messages.slice(-2).map(m => m.content).join(' ').toLowerCase();
  const hasThanksAtEnd = /\b(thank you|thanks|appreciate you|grateful)\b/i.test(lastMessages);
  
  details.push({
    element: 'T - Thank the patient',
    maxPoints: 5,
    earnedPoints: hasThanksAtEnd ? 5 : 0,
    explanation: hasThanksAtEnd 
      ? 'Thanked patient at end of conversation - creates positive closure' 
      : 'Did not thank patient at end - always end encounters with gratitude',
    found: hasThanksAtEnd
  });
  
  if (hasThanksAtEnd) {
    score += 5;
    strengths.push('Thanked patient at conversation end');
  } else {
    improvements.push('End the conversation by thanking the patient');
  }

  return { score, strengths, improvements, missed, details };
}

function analyzeCARE(allText: string, messages: Message[]) {
  let score = 0;
  const strengths: string[] = [];
  const improvements: string[] = [];
  const missed: string[] = [];
  const details: ScoringDetail[] = [];

  // Compassion (6 points)
  const compassionIndicators = /\b(understand|care|here to help|support|with you|concern|important to me)\b/i.test(allText);
  details.push({
    element: 'C - Compassion',
    maxPoints: 6,
    earnedPoints: compassionIndicators ? 6 : 0,
    explanation: compassionIndicators 
      ? 'Expressed genuine care and support for patient wellbeing' 
      : 'Missing compassionate language like "I care" or "I\'m here to help"',
    found: compassionIndicators
  });
  if (compassionIndicators) {
    score += 6;
    strengths.push('Demonstrated compassion throughout conversation');
  } else {
    improvements.push('Show more compassion by expressing genuine care for patient\'s wellbeing');
  }

  // Accountability (6 points)
  const accountabilityIndicators = /\b(i will|we will|i'll make sure|follow up|i'll|responsibility|ensure)\b/i.test(allText);
  details.push({
    element: 'A - Accountability',
    maxPoints: 6,
    earnedPoints: accountabilityIndicators ? 6 : 0,
    explanation: accountabilityIndicators 
      ? 'Took ownership by committing to specific actions ("I will..." statements)' 
      : 'Did not commit to actions - use "I will" to show accountability',
    found: accountabilityIndicators
  });
  if (accountabilityIndicators) {
    score += 6;
    strengths.push('Took accountability for patient care and follow-through');
  } else {
    improvements.push('Take ownership by committing to specific actions');
  }

  // Respect (7 points)
  const respectIndicators = allText.match(/\b(please|thank you|appreciate|respect|understand)\b/gi);
  const questionCount = (allText.match(/\?/g) || []).length;
  
  const respectPoints = ((respectIndicators && respectIndicators.length >= 2) ? 4 : 0) + (questionCount >= 2 ? 3 : 0);
  details.push({
    element: 'R - Respect',
    maxPoints: 7,
    earnedPoints: respectPoints,
    explanation: `${(respectIndicators && respectIndicators.length >= 2) ? `✓ Used respectful language (${respectIndicators.length} instances, 4 pts)` : '✗ Limited respectful language (0/4 pts)'} | ${questionCount >= 2 ? `✓ Asked questions to respect autonomy (${questionCount} questions, 3 pts)` : '✗ Few questions asked (0/3 pts)'}`,
    found: respectPoints > 0
  });
  
  if (respectIndicators && respectIndicators.length >= 2) {
    score += 4;
    strengths.push('Used respectful language consistently');
  } else {
    improvements.push('Use more respectful, courteous language');
  }
  
  if (questionCount >= 2) {
    score += 3;
    strengths.push('Asked for patient input and preferences');
  } else {
    missed.push('Asking more questions to respect patient autonomy');
  }

  // Excellence (6 points)
  const checkUnderstanding = /\b(make sense|understand|questions|concerns|unclear|clear)\b/i.test(allText);
  const thoroughness = /\b(let me explain|important to know|what this means|because|specifically)\b/i.test(allText);
  
  const excellencePoints = (checkUnderstanding ? 3 : 0) + (thoroughness ? 3 : 0);
  details.push({
    element: 'E - Excellence',
    maxPoints: 6,
    earnedPoints: excellencePoints,
    explanation: `${checkUnderstanding ? '✓ Checked understanding (3 pts)' : '✗ Did not verify patient understanding (0/3 pts)'} | ${thoroughness ? '✓ Thorough explanations (3 pts)' : '✗ Limited detail in explanations (0/3 pts)'}`,
    found: excellencePoints > 0
  });
  
  if (checkUnderstanding) {
    score += 3;
    strengths.push('Checked for patient understanding');
  } else {
    improvements.push('Check patient understanding by asking "What questions do you have?"');
  }
  
  if (thoroughness) {
    score += 3;
    strengths.push('Provided thorough explanations');
  }

  return { score, strengths, improvements, missed, details };
}

function analyzeWOW(allText: string, lastMessage: string, messages: Message[]) {
  let score = 0;
  const strengths: string[] = [];
  const improvements: string[] = [];
  const missed: string[] = [];
  const details: ScoringDetail[] = [];

  // Thorough explanations (6 points)
  const avgMessageLength = messages.reduce((sum, m) => sum + m.content.length, 0) / messages.length;
  const isThorough = avgMessageLength > 100;
  details.push({
    element: 'Thorough explanations',
    maxPoints: 6,
    earnedPoints: isThorough ? 6 : 0,
    explanation: isThorough 
      ? `Average message length ${Math.round(avgMessageLength)} chars - detailed responses` 
      : `Average message length ${Math.round(avgMessageLength)} chars - aim for 100+ for thorough explanations`,
    found: isThorough
  });
  if (isThorough) {
    score += 6;
    strengths.push('Provided thorough, detailed responses');
  } else {
    improvements.push('Provide more thorough explanations to ensure clarity');
  }

  // Personal touches (6 points)
  const personalTouches = /\b(your|you're|you've|your situation|for you|with you)\b/gi;
  const personalCount = (allText.match(personalTouches) || []).length;
  
  const isPersonalized = personalCount >= 5;
  details.push({
    element: 'Personal touches',
    maxPoints: 6,
    earnedPoints: isPersonalized ? 6 : 0,
    explanation: isPersonalized 
      ? `Used personalized language ${personalCount} times (your, you're, etc.)` 
      : `Only ${personalCount} personalized words - use "your" and "you" more to connect`,
    found: isPersonalized
  });
  
  if (isPersonalized) {
    score += 6;
    strengths.push('Personalized communication for the patient');
  } else {
    improvements.push('Use more personalized language to connect with patient');
  }

  // Check understanding (7 points)
  const checkPhrases = /\b(make sense|questions|concerns|understand|unclear|what questions do you have)\b/i.test(allText);
  const encourageQuestions = /\b(feel free to ask|any questions|don't hesitate|happy to answer)\b/i.test(allText);
  
  const understandingPoints = (checkPhrases ? 4 : 0) + (encourageQuestions ? 3 : 0);
  details.push({
    element: 'Check understanding',
    maxPoints: 7,
    earnedPoints: understandingPoints,
    explanation: `${checkPhrases ? '✓ Asked if things make sense (4 pts)' : '✗ Did not check understanding (0/4 pts)'} | ${encourageQuestions ? '✓ Encouraged questions (3 pts)' : '✗ Did not invite questions (0/3 pts)'}`,
    found: understandingPoints > 0
  });
  
  if (checkPhrases) {
    score += 4;
    strengths.push('Checked for understanding (WOW micro-skill)');
  } else {
    improvements.push('Ask "What questions do you have?" to check understanding');
  }
  
  if (encourageQuestions) {
    score += 3;
    strengths.push('Encouraged patient to ask questions');
  }

  // Closure and next steps (6 points)
  const hasNextSteps = /\b(next|follow up|will contact|schedule|appointment|come back|call|reach out)\b/i.test(lastMessage);
  const hasSummary = /\b(so|to summarize|in summary|what we've discussed|plan is)\b/i.test(allText);
  
  const closurePoints = (hasNextSteps ? 3 : 0) + (hasSummary ? 3 : 0);
  details.push({
    element: 'Closure & next steps',
    maxPoints: 6,
    earnedPoints: closurePoints,
    explanation: `${hasNextSteps ? '✓ Outlined next steps (3 pts)' : '✗ No follow-up plan mentioned (0/3 pts)'} | ${hasSummary ? '✓ Summarized conversation (3 pts)' : '✗ No summary provided (0/3 pts)'}`,
    found: closurePoints > 0
  });
  
  if (hasNextSteps) {
    score += 3;
    strengths.push('Clearly outlined next steps');
  } else {
    improvements.push('Clearly state next steps and follow-up plan');
  }
  
  if (hasSummary) {
    score += 3;
    strengths.push('Summarized key points for patient');
  } else {
    missed.push('Summarizing the conversation and care plan');
  }

  return { score, strengths, improvements, missed, details };
}

function generateOverallAssessment(score: number, patient: PatientProfile): string {
  if (score >= 90) {
    return `Excellent communication! You demonstrated strong mastery of S.T.A.R.T., H.E.A.R.T., C.A.R.E., and WOW frameworks. Your approach was patient-centered, empathetic, and thorough. The patient would likely feel heard, understood, and confident in their care plan.`;
  } else if (score >= 75) {
    return `Good communication overall. You applied several key principles from the frameworks effectively. With some refinement in areas like empathy expression or checking understanding, you could achieve excellent patient interactions. The patient would likely feel generally satisfied with the encounter.`;
  } else if (score >= 60) {
    return `Your communication needs improvement. While you covered some basics, you missed opportunities to build rapport, demonstrate empathy, and ensure patient understanding. Focus on using the S.T.A.R.T. and H.E.A.R.T. frameworks more consistently to improve patient experience.`;
  } else {
    return `Significant improvement needed. The patient likely left feeling unheard or confused. Review the S.T.A.R.T., H.E.A.R.T., C.A.R.E., and WOW frameworks carefully and practice incorporating greeting, empathy, active listening, and clear explanations into every patient interaction.`;
  }
}
