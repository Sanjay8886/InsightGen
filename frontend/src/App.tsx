import React, { useState, useRef, useEffect, FormEvent } from 'react';
import { Menu, Zap, Upload, FileText, Send, X, Plus, Mic, CheckCircle, AlertTriangle, Loader2, Code, Table } from 'lucide-react';

// --- Configuration ---
const BACKEND_URL = 'https://insight-gen-api-production.up.railway.app'

Export to Sheets
2';

// --- Global Style Reset Component (Fixes 100vh overflow issue) ---
const GlobalStyleReset = () => (
    <style>
{`
    /* CRITICAL FIX: Reset host page */
    html, body, #root { 
        margin: 0 !important;
        padding: 0 !important;
        height: 100%;
        width: 100%;
        overflow: hidden; 
        box-sizing: border-box;
        font-family: 'Inter', sans-serif;
        background-color: #fff;
        color: #111827;
    }

    /* Container makes table responsive */
    .data-table-container {
        width: 100%;
        margin-top: 1rem;
        border-radius: 0.5rem;
        overflow-x: auto;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
        border: 1px solid #E5E7EB;
        background-color: #ffffff;
    }

    /* Table base */
    .data-table-container table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.875rem;
        min-width: 600px; /* prevent squishing on small screens */
    }

    /* Table header */
    .data-table-container th {
        background-color: #F9FAFB;
        font-weight: 600;
        color: #374151;
        text-align: left;
        padding: 0.75rem 0.5rem;
        border-bottom: 2px solid #E5E7EB;
        white-space: nowrap;
    }

    /* Table cells */
    .data-table-container td {
        padding: 0.75rem 0.5rem;
        border-bottom: 1px solid #E5E7EB;
        vertical-align: top;
        color: #111827;
    }

    /* Zebra striping */
    .data-table-container tr:nth-child(even) {
        background-color: #F9FAFB;
    }

    /* Hover effect */
    .data-table-container tr:hover {
        background-color: #F3F4F6;
    }

    /* Remove last row border */
    .data-table-container tr:last-child td {
        border-bottom: none;
    }

    /* Loader animation */
    @keyframes spin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
`}
</style>

);

// --- Helper component for responsive pure-inline styling ---
const useResponsiveStyle = () => {
    const [isMobile, setIsMobile] = useState(window.innerWidth < 1024); 

    useEffect(() => {
        const handleResize = () => {
            setIsMobile(window.innerWidth < 1024);
        };
        window.addEventListener('resize', handleResize);
        return () => window.removeEventListener('remove', handleResize);
    }, []);

    return isMobile;
};

// --- Custom Modal Component (Replaces alert) ---
interface ModalProps {
    isOpen: boolean;
    onClose: () => void;
    title: string;
    message: string;
    status: 'success' | 'error' | 'info';
    fileName?: string | null;
}

const CustomModal: React.FC<ModalProps> = ({ isOpen, onClose, title, message, status, fileName }) => {
    if (!isOpen) return null;

    const Icon = status === 'success' ? CheckCircle : AlertTriangle;
    const color = status === 'success' ? '#10B981' : (status === 'error' ? '#EF4444' : '#F59E0B');
    const bgColor = status === 'success' ? '#ECFDF5' : (status === 'error' ? '#FEF2F2' : '#FFFBEB');

    return (
        <div style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: 'rgba(0, 0, 0, 0.4)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 100
        }}>
            <div style={{
                backgroundColor: 'white',
                padding: '2rem',
                borderRadius: '1rem',
                maxWidth: '400px',
                width: '90%',
                boxShadow: '0 10px 25px rgba(0, 0, 0, 0.2)',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
            }}>
                <div style={{ 
                    backgroundColor: bgColor, 
                    padding: '1rem', 
                    borderRadius: '50%', 
                    marginBottom: '1rem',
                    border: `2px solid ${color}`
                }}>
                    <Icon style={{ width: 32, height: 32, color }} />
                </div>

                <h3 style={{ fontSize: '1.25rem', fontWeight: 'bold', marginBottom: '0.5rem', color: '#1F2937' }}>{title}</h3>
                
                {fileName && status === 'success' && (
                    <p style={{ fontSize: '0.875rem', color: '#4B5563', marginBottom: '1rem', fontWeight: '600' }}>
                        File: **{fileName}**
                    </p>
                )}

                <p style={{ textAlign: 'center', color: '#4B5563', marginBottom: '1.5rem', whiteSpace: 'pre-wrap' }}>{message}</p>
                
                <button 
                    onClick={onClose}
                    style={{
                        backgroundColor: '#4F46E5', 
                        color: 'white', 
                        padding: '0.65rem 1.5rem', 
                        borderRadius: '0.5rem', 
                        border: 'none', 
                        fontWeight: '600',
                        cursor: 'pointer',
                        transition: 'background-color 0.15s'
                    }}
                    onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#4338CA'}
                    onMouseLeave={(e) => e.currentTarget.style.backgroundColor = '#4F46E5'}
                >
                    OK
                </button>
            </div>
        </div>
    );
};


// --- Main Content Area (Mimicking Gemini's look) ---
interface AnalysisResponse {
    summary: string;
    sql_query?: string;
    table_html?: string;
    chart_data?: any;
    status: 'success' | 'error';
    error?: string;
}

interface ChatMessage {
    type: 'user' | 'agent';
    text: string; // Used for general conversation or error text
    summary?: string;
    sql_query?: string;
    table_html?: string;
}

interface MainContentAreaProps {
    analysisActive: boolean; 
    fileName: string | null;
    resetChat: React.MutableRefObject<(() => void) | null>;
    isMobile: boolean;
    openModal: (title: string, message: string, status: 'success' | 'error' | 'info') => void;
}

const MainContentArea: React.FC<MainContentAreaProps> = ({ analysisActive, fileName, resetChat, isMobile, openModal }) => {
  const [query, setQuery] = useState('');
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    resetChat.current = () => setMessages([]);
  }, [resetChat]);

  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages]);

  // --- API Utility Function for Pre-Analysis Conversation (Uses Gemini API directly) ---
  const callGeminiApiWithRetry = async (
      userQuery: string, 
      systemInstruction: string,
      maxRetries = 3
  ): Promise<string> => {
      // FIX: Changed API key retrieval to use the mandated empty string for runtime injection.
      const apiKey = import.meta.env.VITE_GEMINI_API_KEY; 
      
      const apiUrl = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key=${apiKey}`;

      const payload = {
          contents: [{ parts: [{ text: userQuery }] }],
          systemInstruction: {
              parts: [{ text: systemInstruction }]
          },
      };

      for (let attempt = 0; attempt < maxRetries; attempt++) {
          try {
              const response = await fetch(apiUrl, {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify(payload)
              });

              if (response.ok) {
                  const result = await response.json();
                  const text = result.candidates?.[0]?.content?.parts?.[0]?.text;
                  if (text) return text;
                  throw new Error("AI content field was empty.");
              } else if (response.status === 429 && attempt < maxRetries - 1) {
                  const delay = Math.pow(2, attempt) * 1000 + Math.random() * 500;
                  await new Promise(resolve => setTimeout(resolve, delay));
              } else {
                  const errorText = await response.text();
                  throw new Error(`API failed with HTTP Status ${response.status}. Details: ${errorText.substring(0, 150)}...`);
              }
          } catch (error) {
              if (attempt === maxRetries - 1 || !(error instanceof Error)) throw error;
          }
      }
      throw new Error('API request failed after all retries.');
  };

  // --- API Utility Function for Data Analysis (Uses Python Backend) ---
  const callAnalysisApi = async (
      userQuery: string,
      currentFileName: string
  ): Promise<AnalysisResponse> => {
      
      const payload = {
          query: userQuery,
          file_name: currentFileName,
      };

      try {
          // Uses the updated BACKEND_URL (port 5001)
          const response = await fetch(`${BACKEND_URL}/query`, { 
              method: 'POST',
              headers: {
                  'Content-Type': 'application/json',
              },
              body: JSON.stringify(payload)
          });

          if (!response.ok) {
              const errorText = await response.text();
              throw new Error(`Backend failed with status ${response.status}. Message: ${errorText.substring(0, 200)}...`);
          }

          const result: AnalysisResponse = await response.json();

          // Check for structured error from backend, even if HTTP status is 200
          if (result.status === 'error') {
              throw new Error(`Analysis Error: ${result.error || 'The backend failed to return a valid analysis.'}`);
          }
          return result;

      } catch (e: any) {
          console.error("Analysis API Error:", e.message);
          return {
              status: 'error',
              summary: "I encountered an error while trying to process your request with the data.",
              error: e.message,
          };
      }
  };


  const handleSend = async (e: FormEvent) => {
    e.preventDefault();
    const userQuery = query.trim();
    if (userQuery === '' || isLoading) return;

    // Add user message immediately
    setMessages(prev => [...prev, { type: 'user', text: userQuery }]);
    setQuery('');
    setIsLoading(true);

    try {
        if (analysisActive && fileName) {
            // Case 1: Analysis is active (file uploaded) - CALL PYTHON BACKEND
            const analysisResult = await callAnalysisApi(userQuery, fileName);

            if (analysisResult.status === 'success') {
                 setMessages(prev => [
                    ...prev,
                    { 
                        type: 'agent', 
                        summary: analysisResult.summary,
                        sql_query: analysisResult.sql_query,
                        table_html: analysisResult.table_html,
                        text: `Here is the analysis based on your query: "${userQuery}".` // Base text for general context
                    }
                ]);
            } else {
                // Analysis API returned an error status
                setMessages(prev => [...prev, { 
                    type: 'agent', 
                    text: `⚠️ **Data Analysis Failed!** The backend reported an error: \n\n${analysisResult.error || 'Unknown error during data processing.'}` 
                }]);
                openModal('Analysis Error', analysisResult.error || 'Failed to get a structured response from the analysis service.', 'error');
            }

        } else {
            // Case 2: Analysis is NOT active (pre-upload conversation) - CALL GEMINI API
            const systemInstruction = "You are the AI Data Agent for an Excel analysis application. Your primary goal is to engage in friendly, general conversation while subtly encouraging the user to upload a data file (Excel/CSV) to begin the main data analysis task. Keep your responses concise, conversational, and always remind the user that your most powerful features are unlocked after they upload their data.";
            
            const agentResponse = await callGeminiApiWithRetry(userQuery, systemInstruction);

            setMessages(prev => [
                ...prev,
                { 
                    type: 'agent', 
                    text: agentResponse
                }
            ]);
        }
    } catch (error) {
        // Handle critical errors (like network failure for Gemini API or unexpected fetch errors)
        const errorMessage = error instanceof Error ? error.message : "An unknown error occurred.";
        console.error("Critical API Error:", errorMessage);
        
        openModal(
            'API Connection Error',
            `I couldn't complete the request. Details: ${errorMessage}`,
            'error'
        );
        
        setMessages(prev => [...prev, { 
            type: 'agent', 
            text: `⚠️ **Connection Error** reported. I couldn't reach the service.` 
        }]);
    } finally {
        setIsLoading(false);
    }
  };
  
  // Styles for the Input Container (the main white box)
  const inputContainerStyle: React.CSSProperties = {
    display: 'flex',
    alignItems: 'center',
    width: '100%',
    maxWidth: '768px', 
    padding: '0.5rem 1rem', 
    backgroundColor: 'white',
    borderRadius: '1.5rem', 
    boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)', 
    margin: '0 auto', 
    transition: 'box-shadow 0.2s',
  };

  const inputStyle: React.CSSProperties = {
      flexGrow: 1,
      padding: '0.25rem 0',
      border: 'none',
      outline: 'none',
      fontSize: '1rem',
      backgroundColor: 'transparent',
      resize: 'none',
      minHeight: '24px', 
      maxHeight: '200px',
      overflowY: 'auto',
      lineHeight: '1.5',
      cursor: isLoading ? 'not-allowed' : 'text',
      color: isLoading ? '#9CA3AF' : '#1F2937',
  };
  
  // Renders the chat history content
  const renderChatHistory = () => (
    // ADDED PADDING TO RIGHT SIDE OF CONTENT AREA
    <div style={{ padding: '3rem 1.5rem 2rem 1.5rem' }}> 
        {messages.map((msg, index) => (
            <div 
                key={index} 
                style={{ 
                    marginBottom: '1rem', 
                    width: 'fit-content', 
                    // Adjusted max-width to consider padding
                    maxWidth: isMobile ? '85%' : '768px', 
                    marginLeft: msg.type === 'user' ? 'auto' : '0', 
                    marginRight: msg.type === 'user' ? '0' : 'auto', 
                    paddingLeft: msg.type === 'agent' ? (isMobile ? '0' : '1.5rem') : '0', // Keep agent message left-aligned
                    paddingRight: msg.type === 'user' ? (isMobile ? '0' : '1.5rem') : '0', // Keep user message right-aligned
                    display: 'flex',
                    justifyContent: msg.type === 'user' ? 'flex-end' : 'flex-start',
                     // Use full width to contain margin/padding logic
                }}
            >
                <div style={{ 
                    maxWidth: '100%', 
                    display: 'flex', 
                    flexDirection: 'column', 
                    alignItems: msg.type === 'user' ? 'flex-end' : 'flex-start' 
                }}>
                    {/* Message Header */}
                    <p style={{ fontWeight: '600', fontSize: '0.875rem', marginBottom: '0.5rem', color: msg.type === 'user' ? '#3730A3' : '#6366F1', textAlign: msg.type === 'user' ? 'right' : 'left' }}>
                        {msg.type === 'user' ? 'You' : 'AI Agent'}
                    </p>

                    {/* Message Content Container */}
                    <div 
                        style={{
                            padding: '1rem', 
                            borderRadius: '0.75rem',
                            backgroundColor: msg.type === 'user' ? '#E0E7FF' : 'white', 
                            color: '#1F2937', 
                            boxShadow: '0 1px 3px 0 rgba(0, 0, 0, 0.05)',
                            border: msg.type === 'agent' ? '1px solid #E5E7EB' : 'none',
                            maxWidth: isMobile ? '100%' : '768px', // Limit message width
                            width: 'fit-content', // Removed duplicate 'width: 100%' property, keeping 'fit-content' for dynamic width based on message length.
                        }}
                    >
                        {/* Primary Text/Error (Always present) */}
                        <p style={{ whiteSpace: 'pre-wrap', marginBottom: (msg.summary || msg.sql_query || msg.table_html) ? '1rem' : '0' }}>
                            {msg.summary || msg.text}
                        </p>

                        {/* SQL Query Box (If present in Agent response) */}
                        {msg.sql_query && (
                            <div style={{ 
                                marginTop: '1rem', 
                                padding: '0.75rem', 
                                backgroundColor: '#F9FAFB', 
                                borderRadius: '0.5rem', 
                                border: '1px solid #D1D5DB', 
                                fontSize: '0.75rem', 
                                color: '#4B5563',
                            }}>
                                <p style={{ fontWeight: '600', color: '#6366F1', display: 'flex', alignItems: 'center', marginBottom: '0.5rem' }}>
                                    <Code style={{ width: 14, height: 14, marginRight: 6 }} />
                                    Generated SQL Query:
                                </p>
                                <pre style={{ 
                                    whiteSpace: 'pre-wrap', 
                                    wordBreak: 'break-all', 
                                    backgroundColor: '#E5E7EB', 
                                    padding: '0.5rem', 
                                    borderRadius: '0.25rem',
                                    overflowX: 'auto'
                                }}>
                                    {msg.sql_query}
                                </pre>
                            </div>
                        )}
                        
                        {/* HTML Table (If present in Agent response) */}
                        {msg.table_html && (
                            <div style={{ 
                                marginTop: '1rem', 
                                padding: '0.5rem', 
                                backgroundColor: 'white', 
                                borderRadius: '0.5rem', 
                                border: '1px solid #E5E7EB',
                            }}>
                                 <p style={{ fontWeight: '600', color: '#6366F1', display: 'flex', alignItems: 'center', marginBottom: '0.5rem' }}>
                                    <Table style={{ width: 14, height: 14, marginRight: 6 }} />
                                    Result Table:
                                </p>
                                <div className="data-table-container" dangerouslySetInnerHTML={{ __html: msg.table_html }} />
                            </div>
                        )}
                    </div>
                </div>
            </div>
        ))}

        {/* Loading Indicator */}
        {isLoading && (
            <div 
                style={{ 
                    marginBottom: '1rem', 
                    padding: '1rem', 
                    borderRadius: '0.75rem',
                    width: 'fit-content', 
                    maxWidth: isMobile ? '85%' : '60%', 
                    marginLeft: '0', 
                    marginRight: 'auto', 
                    backgroundColor: 'white', 
                    boxShadow: '0 1px 3px 0 rgba(0, 0, 0, 0.05)',
                    border: '1px solid #E5E7EB',
                    display: 'flex',
                    alignItems: 'center',
                    color: '#6366F1'
                }}
            >
                <Loader2 style={{ width: 16, height: 16, marginRight: 8, animation: 'spin 1s linear infinite' }} />
                <p style={{ fontSize: '0.875rem' }}>AI Agent is thinking...</p>
                <style>{`
                    @keyframes spin {
                        from { transform: rotate(0deg); }
                        to { transform: rotate(360deg); }
                    }
                `}</style>
            </div>
        )}
        <div ref={messagesEndRef} /> {/* For scroll-to-bottom */}
    </div>
  );
  
  // Renders the welcome screen content
  const renderWelcomeScreen = () => (
    <div style={{ 
        flexGrow: 1, 
        display: 'flex', 
        flexDirection: 'column', 
        alignItems: 'center',
        padding: '3rem 1.5rem 2rem 1.5rem', // ADDED PADDING TO RIGHT SIDE OF CONTENT AREA
        height: '100%', 
    }}>
        <div style={{ 
            flexGrow: 1, 
            display: 'flex', 
            flexDirection: 'column', 
            alignItems: 'center',
            justifyContent: 'center',
            paddingBottom: '10vh' 
        }}>
            <h1 style={{ fontSize: '2.25rem', fontWeight: 'normal', color: '#1F2937', marginBottom: '2rem' }}>
                Hello
            </h1>
            
            <p style={{ fontSize: '1.125rem', color: '#4B5563', textAlign: 'center', maxWidth: '600px', lineHeight: '1.6' }}>
                Welcome to the **AI Data Agent**. Please upload your **Excel file** from the sidebar to begin asking complex business questions about your data.
            </p>
        </div>
        <div ref={messagesEndRef} /> {/* For scroll-to-bottom */}
    </div>
  );


  // Bottom input area, which is always visible
  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
        
        {/* 1. CONDITIONAL HEADER (Fixed Height) */}
        {messages.length > 0 && analysisActive && (
            <div style={{ 
                flexShrink: 0, 
                padding: '1rem 1.5rem', 
                borderBottom: '1px solid #E5E7EB', 
                backgroundColor: 'white',
                display: 'flex',
                justifyContent: 'center',
            }}>
                <h2 style={{ fontSize: '1.125rem', fontWeight: '600', color: '#1F2937', textAlign: 'center', maxWidth: '768px' }}>
                    Data Analysis: {fileName}
                </h2>
            </div>
        )}

        {/* 2. SCROLLABLE CONTENT AREA */}
        <div style={{ flexGrow: 1, overflowY: 'auto', minHeight: 0, width: '100%' }}>
            {messages.length > 0 ? renderChatHistory() : renderWelcomeScreen()}
        </div>

        {/* 3. FIXED INPUT BOX (Footer) */}
        <div style={{ 
            padding: '0.75rem 1.5rem', 
            display: 'flex', 
            justifyContent: 'center',
            alignItems: 'center', 
            flexShrink: 0, 
            borderTop: (messages.length > 0 && !analysisActive) ? '1px solid #E5E7EB' : 'none', 
            backgroundColor: 'inherit',
            width: '100%',
        }}>
            <form onSubmit={handleSend} style={inputContainerStyle}>
                
                {/* Placeholder/Icon on Left */}
                <button 
                    type="button" 
                    style={{ padding: '0.5rem', color: '#6366F1', border: 'none', background: 'none', cursor: 'pointer' }}
                >
                    <Plus style={{ width: 24, height: 24 }} />
                </button>

                {/* Text Input Area */}
                <textarea
                    rows={1}
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    onKeyDown={(e) => {
                        if (e.key === 'Enter' && !e.shiftKey) {
                            e.preventDefault();
                            handleSend(e);
                        }
                    }}
                    placeholder={isLoading ? "Waiting for AI response..." : "Ask your question about the data..."} 
                    style={inputStyle}
                    disabled={isLoading}
                />
                
                {/* Send and Mic Icons on Right */}
                <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
                    <button
                        type="submit"
                        disabled={query.trim() === '' || isLoading}
                        style={{ 
                            padding: '0.5rem', 
                            color: (query.trim() && !isLoading) ? '#6366F1' : '#D1D5DB', 
                            border: 'none', 
                            background: 'none', 
                            cursor: (query.trim() && !isLoading) ? 'pointer' : 'default',
                            transition: 'color 0.15s'
                        }}
                    >
                        {isLoading ? <Loader2 style={{ width: 24, height: 24, animation: 'spin 1s linear infinite' }} /> : <Send style={{ width: 24, height: 24 }} />}
                        
                    </button>
                    <Mic style={{ width: 24, height: 24, color: '#9CA3AF' }} />
                </div>
            </form>
        </div>
    </div>
  );
};

// Main App Component
const App: React.FC = () => {
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [file, setFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false); // NEW state for upload status
  const fileInputRef = useRef<HTMLInputElement>(null);
  const isMobile = useResponsiveStyle(); 
  const resetChatRef = useRef<(() => void) | null>(null);

  // Modal State
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [modalContent, setModalContent] = useState({ title: '', message: '', status: 'info' as 'success' | 'error' | 'info' });

  const openModal = (title: string, message: string, status: 'success' | 'error' | 'info') => {
      setModalContent({ title, message, status });
      setIsModalOpen(true);
  };


  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      // Basic validation for Excel files
      if (selectedFile.name.endsWith('.xlsx') || selectedFile.name.endsWith('.xls') || selectedFile.name.endsWith('.csv')) {
        setFile(selectedFile);
      } else {
        openModal('Unsupported File Type', 'Please select a file with an .xlsx, .xls, or .csv extension.', 'error');
        // Clear the input value
        if (fileInputRef.current) fileInputRef.current.value = "";
        setFile(null);
      }
    }
  };

  const handleUploadClick = async () => {
    if (!file) {
      openModal(
          'Missing File',
          'Please select a data file (.xlsx, .xls, or .csv) before attempting to upload.',
          'error'
      );
      return;
    }
    
    setIsUploading(true);
    
    // --- ACTUAL FILE UPLOAD LOGIC ---
    try {
        const formData = new FormData();
        formData.append('file', file);

        // Uses the updated BACKEND_URL (port 5001)
        const response = await fetch(`${BACKEND_URL}/upload`, { 
            method: 'POST',
            body: formData,
        });

        const result = await response.json();

        if (response.ok && result.status === 'success') {
             openModal(
                'Upload Complete', 
                `File **${file.name}** was successfully processed and the database is ready for querying.`, 
                'success'
            );
            // File state remains set, analysis is now active
            
        } else {
            // Handle errors from the backend response structure
             openModal(
                'Upload Failed',
                `The server could not process the file. Error: ${result.error || 'Unknown server error.'}`,
                'error'
            );
            setFile(null);
        }
        
    } catch (e: any) {
         // Handle network or fetch errors
        openModal(
            'Connection Error',
            `Could not connect to the backend server at ${BACKEND_URL}. Please ensure the Python server is running. Error: ${e.message}`,
            'error'
        );
        setFile(null);
    } finally {
        setIsUploading(false);
    }
    // --- END ACTUAL FILE UPLOAD LOGIC ---
  };
  
  const startNewChat = () => {
      setFile(null);
      if (fileInputRef.current) {
          fileInputRef.current.value = ""; // Reset the input file input value
      }
      if (resetChatRef.current) {
          resetChatRef.current(); // Clear chat history using the ref function
      }
      openModal(
        'New Session',
        'Analysis context cleared. Please upload a new data file to begin a fresh data chat.',
        'info'
      );
  };

  const isAnalysisActive = !!file;
  
  // --- PURE INLINE CSS STYLES ---

  const appContainerStyle: React.CSSProperties = {
    display: 'flex',
    minHeight: '100vh', 
    height: '100%', 
    width: '100vw', 
    backgroundColor: '#F3F4F6', // gray-100 (light background)
    fontFamily: 'Inter, sans-serif',
    overflow: 'hidden', 
    flexDirection: isMobile ? 'column' : 'row',
  };

  const sidebarStyle: React.CSSProperties = {
    width: isMobile ? '100%' : '280px', 
    height: '100%', 
    backgroundColor: 'white',
    boxShadow: isMobile ? '0 4px 6px rgba(0, 0, 0, 0.1)' : '4px 0 12px rgba(0, 0, 0, 0.1)', 
    padding: isMobile ? '1.5rem' : '0 1.5rem 1.5rem 1.5rem', 
    paddingTop: isMobile ? '1.5rem' : '0',
    transition: 'transform 0.3s ease-in-out',
    flexShrink: 0,
    zIndex: 20,
    position: isMobile ? 'fixed' : 'relative',
    transform: isMobile && !isSidebarOpen ? 'translateX(-100%)' : 'translateX(0)',
    top: 0,
    left: 0,
    overflowY: 'auto', 
    display: 'flex',
    flexDirection: 'column', 
  };
  
  const mainContentWrapperStyle: React.CSSProperties = {
    flexGrow: 1,
    display: 'flex', 
    flexDirection: 'column', 
    minWidth: 0, 
    minHeight: 0, 
    height: '100%', 
    flex: 1,
  };
  
  const newButtonStyle: React.CSSProperties = {
    width: '100%',
    marginBottom: '1.5rem',
    padding: '0.65rem 1rem',
    backgroundColor: '#EEF2FF', // indigo-50
    color: '#3730A3', // indigo-700
    fontWeight: '600',
    borderRadius: '0.75rem',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    border: '1px solid #C7D2FE', // indigo-200
    cursor: 'pointer',
    transition: 'background-color 0.15s, box-shadow 0.15s',
    whiteSpace: 'nowrap',
  };
  
  const uploadButtonStyle: React.CSSProperties = {
    width: '100%',
    padding: '0.65rem 1rem',
    backgroundColor: (file && !isUploading) ? '#4F46E5' : '#A5B4FC', 
    color: 'white',
    fontWeight: '600',
    borderRadius: '0.75rem',
    boxShadow: '0 2px 4px 0 rgba(0, 0, 0, 0.1)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    border: 'none',
    cursor: (file && !isUploading) ? 'pointer' : 'default',
    marginTop: '0.75rem',
    transition: 'background-color 0.15s',
  };
  
  const statusAreaStyle: React.CSSProperties = {
    padding: '1rem',
    borderRadius: '0.75rem',
    borderWidth: '1px',
    backgroundColor: isAnalysisActive ? '#ECFDF5' : '#FFFBEB', 
    borderColor: isAnalysisActive ? '#A7F3D0' : '#FDE68A', 
    color: '#1F2937', 
    marginTop: 'auto', 
    display: 'flex', 
    flexDirection: 'column',
    alignItems: 'flex-start',
  };

  // --- RENDERING ---

  return (
    // Inject the style reset before the main app content
    <>
        <GlobalStyleReset />
        <div style={appContainerStyle}>
          
          {/* 1. Sidebar for Data Source Control */}
          <div style={sidebarStyle}>
            
            {/* Header and Close Button */}
            <div style={{ 
                display: 'flex', 
                alignItems: 'center', 
                justifyContent: 'space-between', 
                padding: '1.5rem 0',
                flexShrink: 0
            }}>
                <h2 style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#1F2937' }}>
                    <Zap style={{ width: 24, height: 24, display: 'inline', marginRight: 8, color: '#4F46E5' }} />
                    AI Data Agent
                </h2>
                {isMobile && (
                    <button 
                        onClick={() => setIsSidebarOpen(false)}
                        style={{ padding: 4, borderRadius: '50%', border: 'none', background: 'none', cursor: 'pointer' }}
                    >
                        <X style={{ width: 24, height: 24, color: '#6B7280' }} />
                    </button>
                )}
            </div>

            {/* New Chat Button */}
            <button
                onClick={startNewChat}
                style={newButtonStyle}
                onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#E0E7FF'}
                onMouseLeave={(e) => e.currentTarget.style.backgroundColor = '#EEF2FF'}
            >
                <Plus style={{ width: 18, height: 18, marginRight: 8 }} />
                New Analysis Session
            </button>
            
            {/* File Input Controls */}
            <div style={{ 
                flexGrow: 1, // Allows file upload area to take vertical space
                borderTop: '1px solid #E5E7EB', 
                paddingTop: '1.5rem',
                marginBottom: '1.5rem'
            }}>
                <label style={{ display: 'block', fontSize: '1rem', fontWeight: '600', color: '#1F2937', marginBottom: '0.75rem' }}>
                    Data Source
                </label>
                
                <input
                    type="file"
                    ref={fileInputRef}
                    onChange={handleFileChange}
                    style={{ 
                        display: 'block', 
                        width: '100%', 
                        fontSize: '0.875rem', 
                        color: '#4B5563',
                        padding: '0.5rem',
                        border: '1px solid #D1D5DB',
                        borderRadius: '0.5rem',
                        marginBottom: '0.5rem'
                    }}
                    accept=".xlsx,.xls,.csv"
                    disabled={isUploading}
                />
                
                <p style={{ fontSize: '0.75rem', color: '#6B7280', marginBottom: '1rem' }}>
                    Accepted formats: .xlsx, .xls, .csv
                </p>

                <button
                    onClick={handleUploadClick}
                    style={{
                        ...uploadButtonStyle,
                        backgroundColor: (file && !isUploading) ? '#4F46E5' : '#D1D5DB',
                        cursor: (file && !isUploading) ? 'pointer' : 'default',
                    }}
                    disabled={!file || isUploading}
                    onMouseEnter={(e) => {
                        if (file && !isUploading) e.currentTarget.style.backgroundColor = '#4338CA';
                    }}
                    onMouseLeave={(e) => {
                        if (file && !isUploading) e.currentTarget.style.backgroundColor = '#4F46E5';
                    }}
                >
                    {isUploading ? (
                        <Loader2 style={{ width: 20, height: 20, marginRight: 8, animation: 'spin 1s linear infinite' }} />
                    ) : (
                        <Upload style={{ width: 20, height: 20, marginRight: 8 }} />
                    )}
                    {isUploading ? 'Processing...' : 'Upload & Analyze Data'}
                </button>
            </div>


            {/* File Status Area (always at the bottom of the sidebar) */}
            <div style={statusAreaStyle}>
                <div style={{ display: 'flex', alignItems: 'center', marginBottom: '0.5rem' }}>
                    <FileText style={{ width: 18, height: 18, marginRight: 8, color: isAnalysisActive ? '#059669' : '#F59E0B' }} />
                    <span style={{ fontWeight: '600', color: isAnalysisActive ? '#059669' : '#F59E0B' }}>
                        Data Status: {isAnalysisActive ? 'Ready' : 'Awaiting Upload'}
                    </span>
                </div>
                <p style={{ fontSize: '0.875rem', color: '#4B5563' }}>
                    {file ? `File Loaded: ${file.name}` : 'No file selected. Upload an Excel/CSV file to start analysis.'}
                </p>
                <p style={{ fontSize: '0.75rem', color: '#6B7280', marginTop: '0.5rem' }}>
                    Backend Target: {BACKEND_URL}
                </p>
            </div>

          </div> {/* End of Sidebar */}

          {/* 2. Main Content Area */}
          <div style={mainContentWrapperStyle}>
            
            {/* Mobile Menu Button */}
            {isMobile && (
                <div style={{ flexShrink: 0, padding: '0.75rem 1.5rem', backgroundColor: 'white', borderBottom: '1px solid #E5E7EB' }}>
                    <button
                        onClick={() => setIsSidebarOpen(true)}
                        style={{ 
                            padding: '0.5rem', 
                            borderRadius: '0.5rem', 
                            backgroundColor: '#F3F4F6', 
                            border: 'none', 
                            cursor: 'pointer',
                            display: 'flex',
                            alignItems: 'center'
                        }}
                    >
                        <Menu style={{ width: 20, height: 20, marginRight: 8, color: '#4F46E5' }} />
                        <span style={{ fontWeight: '500', color: '#1F2937' }}>Open Sidebar</span>
                    </button>
                </div>
            )}
            
            <MainContentArea
                analysisActive={isAnalysisActive}
                fileName={file ? file.name : null}
                resetChat={resetChatRef}
                isMobile={isMobile}
                openModal={openModal}
            />

          </div> {/* End of Main Content Area */}

          {/* 3. Modal */}
          <CustomModal
                isOpen={isModalOpen}
                onClose={() => setIsModalOpen(false)}
                title={modalContent.title}
                message={modalContent.message}
                status={modalContent.status}
                fileName={file?.name}
          />
        </div> {/* End of App Container */}
    </>
  );
};
export default App;

