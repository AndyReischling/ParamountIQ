import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Alert, AlertDescription } from './ui/alert';
import { Key, AlertCircle, CheckCircle2 } from 'lucide-react';
import { setStoredApiKey, hasValidApiKey } from '../utils/apiKeyManager';

interface ApiKeySetupProps {
  onComplete: () => void;
}

export function ApiKeySetup({ onComplete }: ApiKeySetupProps) {
  const [apiKey, setApiKey] = useState('');
  const [error, setError] = useState('');
  const [success, setSuccess] = useState(false);

  const handleSave = () => {
    setError('');
    setSuccess(false);

    if (!apiKey.trim()) {
      setError('Please enter an API key');
      return;
    }

    if (!apiKey.startsWith('sk-')) {
      setError('OpenAI API keys should start with "sk-" or "sk-proj-"');
      return;
    }

    if (apiKey.length < 20) {
      setError('API key seems too short. Please check and try again.');
      return;
    }

    // Store the API key
    setStoredApiKey(apiKey.trim());
    setSuccess(true);

    // Wait a moment to show success message, then complete
    setTimeout(() => {
      onComplete();
    }, 1000);
  };

  const handleSkip = () => {
    // Store a placeholder to allow the app to run without AI
    setStoredApiKey('DEMO_MODE');
    onComplete();
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-teal-50 flex items-center justify-center p-4">
      <Card className="max-w-2xl w-full">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Key className="w-5 h-5 text-accent" />
            OpenAI API Key Required
          </CardTitle>
          <CardDescription>
            This application uses OpenAI's GPT models to simulate realistic patient interactions and provide detailed feedback.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <Alert>
            <AlertCircle className="h-4 w-4" />
            <AlertDescription className="text-xs">
              <strong>Get your API key:</strong>
              <ol className="list-decimal list-inside mt-2 space-y-1">
                <li>Visit <a href="https://platform.openai.com/api-keys" target="_blank" rel="noopener noreferrer" className="text-accent underline">platform.openai.com/api-keys</a></li>
                <li>Sign in or create an account</li>
                <li>Click "Create new secret key"</li>
                <li>Copy the key and paste it below</li>
              </ol>
              <p className="mt-2 text-amber-700">
                <strong>Security:</strong> Your API key is stored locally in your browser and never sent to any server except OpenAI.
              </p>
            </AlertDescription>
          </Alert>

          <div className="space-y-2">
            <label htmlFor="apiKey" className="text-sm text-gray-700">
              OpenAI API Key
            </label>
            <Input
              id="apiKey"
              type="password"
              placeholder="sk-proj-..."
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSave()}
              className="font-mono text-sm"
            />
            <p className="text-xs text-gray-600">
              Your key starts with "sk-" or "sk-proj-"
            </p>
          </div>

          {error && (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription className="text-xs">{error}</AlertDescription>
            </Alert>
          )}

          {success && (
            <Alert className="bg-green-50 text-green-900 border-green-200">
              <CheckCircle2 className="h-4 w-4 text-green-600" />
              <AlertDescription className="text-xs">
                API key saved successfully! Launching simulator...
              </AlertDescription>
            </Alert>
          )}

          <div className="flex gap-3 pt-4">
            <Button 
              onClick={handleSave} 
              className="flex-1 bg-accent hover:bg-accent/90"
              disabled={success}
            >
              Save & Continue
            </Button>
            <Button 
              onClick={handleSkip} 
              variant="outline"
              className="border-gray-300"
            >
              Skip (Demo Mode)
            </Button>
          </div>

          <Alert className="bg-amber-50 border-amber-200">
            <AlertCircle className="h-4 w-4 text-amber-600" />
            <AlertDescription className="text-xs text-amber-900">
              <strong>Demo Mode:</strong> If you skip, the app will use fallback responses instead of AI-generated patient interactions. For the best experience, add your API key.
            </AlertDescription>
          </Alert>
        </CardContent>
      </Card>
    </div>
  );
}
