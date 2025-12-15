import { useState, useEffect } from 'react';
import { SPSSimulator } from './components/SPSSimulator';
import { ApiKeySetup } from './components/ApiKeySetup';
import { Button } from './components/ui/button';
import { RefreshCw, Key } from 'lucide-react';
import { isConfigured, clearStoredApiKey } from './utils/apiKeyManager';

export default function App() {
  const [key, setKey] = useState(0);
  const [showApiKeySetup, setShowApiKeySetup] = useState(false);

  useEffect(() => {
    // Only show setup if no API key has been configured at all
    if (!isConfigured()) {
      setShowApiKeySetup(true);
    }
  }, []);

  const handleReset = () => {
    setKey(prev => prev + 1);
  };

  const handleApiKeyComplete = () => {
    setShowApiKeySetup(false);
  };

  const handleChangeApiKey = () => {
    clearStoredApiKey();
    setShowApiKeySetup(true);
  };

  if (showApiKeySetup) {
    return <ApiKeySetup onComplete={handleApiKeyComplete} />;
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-teal-50 p-4">
      <div className="max-w-5xl mx-auto py-8">
        <div className="text-center mb-8">
          <h1 className="mb-2 text-gray-900">
            Standardized Patient Simulator (SPS)
          </h1>
          <p className="text-gray-700">
            Clinical education tool for practicing history taking, physical examination, and clinical reasoning
          </p>
          <p className="text-sm text-gray-600 mt-2">
            Each case is a realistic patient encounter with comprehensive evaluation
          </p>
        </div>

        <div className="mb-4 flex justify-between items-center">
          <Button 
            onClick={handleChangeApiKey} 
            variant="ghost" 
            size="sm"
            className="gap-2 text-gray-600 hover:text-gray-900"
          >
            <Key className="w-4 h-4" />
            Change API Key
          </Button>
          <Button onClick={handleReset} variant="outline" className="gap-2 border-primary text-primary hover:bg-primary/10">
            <RefreshCw className="w-4 h-4" />
            New Patient Case
          </Button>
        </div>

        <SPSSimulator key={key} />
      </div>
    </div>
  );
}
