import { PatientProfile } from './workingPatientResponses';

interface Message {
  role: 'patient' | 'provider';
  content: string;
  timestamp: Date;
}

interface PatientResponse {
  content: string;
  shouldEnd: boolean;
}

interface PatientContext {
  condition: string;
  symptoms: string[];
  timeline: string;
  severity: string;
  treatments: string[];
  testResults: Map<string, string>;
  fears: string[];
  discussedTopics: Set<string>;
  providerName: string;
  trustLevel: number;
}

const patientContexts = new Map<string, PatientContext>();

// COMPLETELY SYNCHRONOUS - NO ASYNC NEEDED
export function getPatientResponse(
  patient: PatientProfile,
  providerMessage: string,
  conversationHistory: Message[],
  turnCount: number,
): PatientResponse {
  
  // Get or create patient context
  const context = getOrCreateContext(patient, conversationHistory);
  
  // Generate response completely synchronously
  const response = generateResponse(
    patient,
    providerMessage,
    context,
    turnCount
  );
  
  // Update what's been discussed
  updateContext(context, providerMessage, response);
  
  return {
    content: response,
    shouldEnd: turnCount >= 7 || /\b(goodbye|take care|see you)\b/i.test(providerMessage)
  };
}

function getOrCreateContext(patient: PatientProfile, history: Message[]): PatientContext {
  if (patientContexts.has(patient.id)) {
    return patientContexts.get(patient.id)!;
  }
  
  const context: PatientContext = {
    condition: patient.concern,
    symptoms: [],
    timeline: '',
    severity: '',
    treatments: [],
    testResults: new Map(),
    fears: [],
    discussedTopics: new Set(),
    providerName: '',
    trustLevel: 0
  };
  
  // Set up condition-specific details based on concern
  if (patient.concern.includes('insurance denied')) {
    context.condition = 'severe chronic headaches/migraines';
    context.symptoms = [
      'severe headaches 4-5 times a week',
      'vision changes - zigzag lines before the pain',
      'throbbing pain on the right side',
      'sensitivity to light and sound',
      'neck pain and stiffness'
    ];
    context.timeline = 'headaches for 6 months, much worse last 2 months';
    context.testResults.set('MRI', 'ordered by neurologist Dr. Chen');
    context.testResults.set('Insurance', 'denied - "not medically necessary"');
    context.testResults.set('Cost', '$3,200 out of pocket');
    context.treatments = [
      'Topamax - too many side effects',
      'propranolol - no improvement', 
      'amitriptyline - made me too drowsy',
      'physical therapy for neck',
      'diet changes and trigger avoidance'
    ];
    context.severity = '8-9 out of 10 during attacks';
    context.fears = [
      'brain tumor or aneurysm being missed',
      "can't afford the MRI",
      'losing my job from missing work'
    ];
  } else if (patient.concern.includes('PSA')) {
    context.symptoms = [
      'getting up 3-4 times a night to urinate',
      'weak urine stream',
      'feeling of incomplete bladder emptying',
      'occasional urgency'
    ];
    context.timeline = 'PSA elevated at checkup 3 weeks ago';
    context.testResults.set('PSA', '6.8 ng/mL');
    context.testResults.set('Previous PSA', '3.2 ng/mL last year');
    context.treatments = ['saw palmetto supplement', 'watchful waiting'];
    context.severity = 'no pain, just urinary symptoms';
    context.fears = ['prostate cancer', 'biopsy side effects', 'treatment complications'];
  } else if (patient.concern.includes('wound infection')) {
    context.symptoms = [
      'redness spreading around incision',
      'yellow-green drainage with odor',
      'incision opening up in one spot',
      'low-grade fever 100-101°F'
    ];
    context.timeline = 'surgery 3 weeks ago, infection started 10 days ago';
    context.treatments = ['Keflex for 7 days - just finished', 'wound cleaning with peroxide'];
    context.severity = 'pain 6-7/10 and getting worse';
    context.fears = ['need another surgery', 'sepsis', 'missing more work'];
  } else if (patient.concern.includes('back pain')) {
    context.symptoms = [
      'sharp stabbing lower back pain',
      'pain shooting down left leg',
      'numbness in left foot',
      "can't stand for more than 10 minutes"
    ];
    context.timeline = 'started 3 months ago lifting a heavy box';
    context.testResults.set('MRI', 'L4-L5 disc herniation');
    context.treatments = [
      'physical therapy for 8 weeks',
      'epidural injection - helped 4 days',
      'ibuprofen 800mg TID'
    ];
    context.severity = '7-8/10 constant, spikes to 10/10';
  } else if (patient.concern.includes('migraine')) {
    context.symptoms = [
      'throbbing right-sided headache',
      'visual auras - flashing lights',
      'severe nausea and vomiting',
      'complete light/sound sensitivity'
    ];
    context.timeline = 'monthly before, now daily for 2 months';
    context.treatments = ['Excedrin Migraine', "borrowed friend's Imitrex"];
    context.severity = '9-10/10, completely debilitating';
    context.fears = ['brain tumor', 'losing job', 'becoming disabled'];
  }
  
  // Check history for provider name
  history.forEach(msg => {
    if (msg.role === 'provider') {
      const nameMatch = msg.content.match(/I'm Dr\. (\w+)|I am Dr\. (\w+)|my name is (\w+)/i);
      if (nameMatch) {
        context.providerName = nameMatch[1] || nameMatch[2] || nameMatch[3];
      }
    }
  });
  
  patientContexts.set(patient.id, context);
  return context;
}

function generateResponse(
  patient: PatientProfile,
  providerMessage: string,
  context: PatientContext,
  turnCount: number
): string {
  
  const message = providerMessage.toLowerCase();
  
  // First response - opening
  if (turnCount === 1 && context.discussedTopics.size === 0) {
    context.discussedTopics.add('opening');
    
    if (patient.concern.includes('insurance denied')) {
      if (patient.emotionalState === 'Frustrated') {
        return "Look, I really need help. My insurance denied the MRI my neurologist ordered for my severe headaches. They say it's 'not medically necessary' but I'm having headaches almost daily! The MRI costs $3,200 and I can't afford that. I don't know what to do.";
      }
      return "I'm here about an insurance denial. My neurologist Dr. Chen ordered an MRI for my severe headaches, but insurance says it's not necessary. I'm really struggling with these headaches and need help figuring out what to do.";
    }
    
    return `I need help with ${context.condition}. ${context.timeline}.`;
  }
  
  // Handle greetings/introductions
  if (/hello|introduce|sit down|mind if/i.test(message)) {
    if (message.includes('introduce')) {
      if (patient.emotionalState === 'Frustrated') {
        return "Go ahead, but I really need help today. I've been dealing with this too long.";
      }
      return "Please do. Thank you for taking the time.";
    }
    if (message.includes('sit')) {
      return "Yes, please sit. Thank you for asking.";
    }
    if (context.providerName) {
      return `Hello Dr. ${context.providerName}. Thank you for seeing me.`;
    }
    return "Hello. Thank you for seeing me about this.";
  }
  
  // Answer symptom questions
  if (/tell me|describe|symptoms?|what.*happening|what.*wrong|what.*going on/i.test(message)) {
    if (!context.discussedTopics.has('symptoms')) {
      context.discussedTopics.add('symptoms');
      
      if (patient.concern.includes('insurance denied')) {
        return "I get severe headaches 4-5 times a week. Before they start, I see these zigzag flashing lights. The pain is usually on the right side, throbbing. Light and sound become unbearable. My neck also gets really stiff. It's been going on for months but getting worse.";
      }
      
      const symptoms = context.symptoms.slice(0, 3).join(', ');
      return `I have ${symptoms}. It's been really difficult to manage.`;
    }
    // Already discussed - add new detail
    const newSymptom = context.symptoms.find(s => !context.discussedTopics.has(s));
    if (newSymptom) {
      context.discussedTopics.add(newSymptom);
      return `I also have ${newSymptom}. That's been concerning too.`;
    }
    return "Like I mentioned, those are the main symptoms affecting me.";
  }
  
  // Timeline questions
  if (/how long|when.*start|when.*began|timeline/i.test(message)) {
    if (!context.discussedTopics.has('timeline')) {
      context.discussedTopics.add('timeline');
      
      if (patient.concern.includes('insurance denied')) {
        return "The headaches started about 6 months ago, but they've gotten much worse the last 2 months. That's when Dr. Chen ordered the MRI. The insurance denial came through last week.";
      }
      
      return context.timeline;
    }
    return `As I said, ${context.timeline}.`;
  }
  
  // Severity questions
  if (/scale|rate|how bad|severity|intense/i.test(message)) {
    if (!context.discussedTopics.has('severity')) {
      context.discussedTopics.add('severity');
      
      if (patient.concern.includes('insurance denied')) {
        return "During a headache, it's 8 or 9 out of 10. Sometimes a full 10 - I literally can't function. I have to lie in a dark room. I've missed so much work. That's why Dr. Chen thinks the MRI is necessary.";
      }
      
      return context.severity;
    }
    return "It's still at that same severity level - hasn't improved at all.";
  }
  
  // Treatment questions
  if (/what.*tried|treatment|medication|what.*done/i.test(message)) {
    if (!context.discussedTopics.has('treatments')) {
      context.discussedTopics.add('treatments');
      
      if (patient.concern.includes('insurance denied')) {
        return "I've tried three preventive medications - Topamax had terrible side effects, propranolol didn't help, amitriptyline made me too drowsy to function. I've done physical therapy for my neck, changed my diet, kept a headache diary. Nothing's really worked. Dr. Chen says we need the MRI to rule out structural causes before trying more aggressive treatments.";
      }
      
      const treatments = context.treatments.slice(0, 2).join(' and ');
      return `I've tried ${treatments}, but nothing has really helped.`;
    }
    return "I've tried everything I mentioned. Nothing seems to work.";
  }
  
  // Insurance/cost questions
  if (/insurance|cost|afford|pay|coverage|denial|appeal/i.test(message)) {
    if (!context.discussedTopics.has('insurance_details')) {
      context.discussedTopics.add('insurance_details');
      return "The insurance company says the MRI isn't 'medically necessary' despite my neurologist ordering it. They want me to try more medications first, but I've already tried three! The MRI costs $3,200 out of pocket. I can't afford that. Have you dealt with insurance appeals before?";
    }
    return "I just don't understand how they can deny something my doctor says I need. What are my options?";
  }
  
  // Emotional/worry questions
  if (/worried|scared|concern|feeling|coping/i.test(message)) {
    if (!context.discussedTopics.has('emotions')) {
      context.discussedTopics.add('emotions');
      
      if (patient.concern.includes('insurance denied')) {
        return "I'm terrified something serious is being missed. What if it's a tumor or aneurysm? But I'm also angry - I pay for insurance every month and when I need it, they deny me. I feel trapped between my health and my finances.";
      }
      
      return `My biggest fear is ${context.fears[0]}. It keeps me up at night.`;
    }
    return "I'm just exhausted from dealing with this. I need it to get better.";
  }
  
  // Plan/next steps
  if (/plan|next|we'll|going to|let's/i.test(message)) {
    if (/appeal|fight|challenge/i.test(message)) {
      return "Yes! How do I appeal this? I'll do whatever paperwork is needed. Should I get a letter from Dr. Chen? I really need this MRI.";
    }
    if (/medication/i.test(message)) {
      return "I'm willing to try something else, but will insurance even cover it? And what if it doesn't work like the others?";
    }
    return "What do you think we should do? I'm open to suggestions but I really need something that will work.";
  }
  
  // Permission questions
  if (/is it okay|do you mind|can i|may i/i.test(message)) {
    return "Sure, that's fine. Whatever helps figure this out.";
  }
  
  // Default follow-up questions
  if (!context.discussedTopics.has('followup')) {
    context.discussedTopics.add('followup');
    
    if (patient.concern.includes('insurance denied')) {
      return "Do you think the insurance company is right? Is the MRI really not necessary? Or are they just trying to avoid paying?";
    }
    
    return "What do you think is causing my symptoms?";
  }
  
  // Final default
  if (patient.emotionalState === 'Frustrated') {
    return "So what's the actual plan here? I need real solutions.";
  }
  
  return "What else do you need to know?";
}

function updateContext(context: PatientContext, providerMessage: string, patientResponse: string): void {
  const combined = (providerMessage + ' ' + patientResponse).toLowerCase();
  
  if (/symptom/i.test(combined)) context.discussedTopics.add('symptoms_mentioned');
  if (/timeline/i.test(combined)) context.discussedTopics.add('timeline_mentioned');
  if (/treatment/i.test(combined)) context.discussedTopics.add('treatment_mentioned');
  if (/insurance/i.test(combined)) context.discussedTopics.add('insurance_mentioned');
  
  // Build trust
  if (/understand|sorry|difficult|must be hard/i.test(providerMessage)) {
    context.trustLevel++;
  }
}