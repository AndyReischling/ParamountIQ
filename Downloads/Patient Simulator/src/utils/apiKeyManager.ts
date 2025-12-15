// API Key Manager for OpenAI
const API_KEY_STORAGE_KEY = 'openai_api_key';

export function getStoredApiKey(): string | null {
  if (typeof window === 'undefined') return null;
  return localStorage.getItem(API_KEY_STORAGE_KEY);
}

export function setStoredApiKey(apiKey: string): void {
  if (typeof window === 'undefined') return;
  localStorage.setItem(API_KEY_STORAGE_KEY, apiKey);
}

export function clearStoredApiKey(): void {
  if (typeof window === 'undefined') return;
  localStorage.removeItem(API_KEY_STORAGE_KEY);
}

export function hasValidApiKey(): boolean {
  const key = getStoredApiKey();
  // Accept DEMO_MODE as valid, or any OpenAI key
  if (!key) return false;
  if (key === 'DEMO_MODE') return true;
  return key.length > 20 && (key.startsWith('sk-') || key.startsWith('sk-proj-'));
}

export function isConfigured(): boolean {
  // Returns true if any API key (including DEMO_MODE) is stored
  const key = getStoredApiKey();
  return !!key;
}
