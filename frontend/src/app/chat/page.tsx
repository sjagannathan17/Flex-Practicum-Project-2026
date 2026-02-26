'use client';

import { useState, useRef, useEffect } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Send, Bot, User, Loader2, Database, Globe, Sparkles, Trash2 } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001';

type SearchMode = 'rag' | 'web' | 'hybrid';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  sources?: Source[];
  mode?: SearchMode;
  timestamp: Date;
}

interface Source {
  company?: string;
  source?: string;
  filing_type?: string;
  fiscal_year?: string;
  similarity?: number;
}

const ANALYST_QUESTIONS = [
  {
    question: "What is the AI/Data Center revenue mix for each company, and how has it changed YoY?",
    category: "AI Investment",
  },
  {
    question: "Compare CapEx guidance across all 5 EMS companies for the current fiscal year",
    category: "CapEx",
  },
  {
    question: "What liquid cooling and power management capabilities are each company developing?",
    category: "AI Infrastructure",
  },
  {
    question: "Which hyperscaler customers are driving AI server demand for EMS companies?",
    category: "Customers",
  },
  {
    question: "What are the gross margin trends for AI/DC vs traditional segments?",
    category: "Financials",
  },
  {
    question: "What manufacturing capacity expansions are planned for AI server production?",
    category: "Operations",
  },
];

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [mode, setMode] = useState<SearchMode>('rag');
  const [sessionId] = useState(() => `session_${Date.now()}`);
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const sendMessage = async (query?: string) => {
    const text = query || input.trim();
    if (!text || isLoading) return;

    const userMsg: Message = {
      id: `user_${Date.now()}`,
      role: 'user',
      content: text,
      timestamp: new Date(),
    };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setIsLoading(true);

    try {
      const res = await fetch(`${API_URL}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: text,
          mode,
          session_id: sessionId,
        }),
      });

      if (!res.ok) throw new Error(`API error: ${res.status}`);
      const data = await res.json();

      const assistantMsg: Message = {
        id: `assistant_${Date.now()}`,
        role: 'assistant',
        content: data.response || data.answer || 'No response received.',
        sources: data.sources || [],
        mode,
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, assistantMsg]);
    } catch (err) {
      const errorMsg: Message = {
        id: `error_${Date.now()}`,
        role: 'assistant',
        content: `Failed to get a response. Make sure the backend is running on port 8001.\n\nError: ${err}`,
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, errorMsg]);
    } finally {
      setIsLoading(false);
      inputRef.current?.focus();
    }
  };

  const clearChat = () => {
    setMessages([]);
  };

  const modeConfig = {
    rag: { icon: Database, label: 'Documents', color: 'bg-blue-100 text-blue-700', desc: 'Search SEC filings' },
    web: { icon: Globe, label: 'Web', color: 'bg-green-100 text-green-700', desc: 'Search the web' },
    hybrid: { icon: Sparkles, label: 'Hybrid', color: 'bg-purple-100 text-purple-700', desc: 'Documents + Web' },
  };

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="border-b bg-white px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-slate-900">AI Chat</h1>
            <p className="text-sm text-slate-500">Ask questions about CapEx, financials, and competitive intelligence</p>
          </div>
          <div className="flex items-center gap-3">
            {/* Mode Toggle */}
            <div className="flex bg-slate-100 rounded-lg p-1">
              {(Object.keys(modeConfig) as SearchMode[]).map((m) => {
                const config = modeConfig[m];
                const Icon = config.icon;
                return (
                  <button
                    key={m}
                    onClick={() => setMode(m)}
                    className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
                      mode === m ? config.color : 'text-slate-500 hover:text-slate-700'
                    }`}
                  >
                    <Icon className="w-3.5 h-3.5" />
                    {config.label}
                  </button>
                );
              })}
            </div>
            {messages.length > 0 && (
              <Button variant="outline" size="sm" onClick={clearChat} className="text-slate-500">
                <Trash2 className="w-4 h-4 mr-1" />
                Clear
              </Button>
            )}
          </div>
        </div>
      </div>

      {/* Messages Area */}
      <ScrollArea className="flex-1 px-6 py-4">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full min-h-[400px] text-center">
            <div className="w-16 h-16 bg-blue-100 rounded-2xl flex items-center justify-center mb-4">
              <Bot className="w-8 h-8 text-blue-600" />
            </div>
            <h2 className="text-xl font-semibold text-slate-800 mb-2">CapEx Intelligence Assistant</h2>
            <p className="text-slate-500 mb-6 max-w-md">
              Ask about capital expenditures, financial data, or competitive analysis across Flex, Jabil, Celestica, Benchmark, and Sanmina.
            </p>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 max-w-2xl">
              {ANALYST_QUESTIONS.map((item, i) => (
                <button
                  key={i}
                  onClick={() => sendMessage(item.question)}
                  className="text-left px-4 py-3 bg-white border border-slate-200 rounded-xl text-sm text-slate-700 hover:bg-blue-50 hover:border-blue-300 transition-all group"
                >
                  <div className="flex items-start gap-2">
                    <span className="px-2 py-0.5 text-xs font-medium bg-slate-100 text-slate-500 rounded group-hover:bg-blue-100 group-hover:text-blue-600 transition-colors">
                      {item.category}
                    </span>
                  </div>
                  <p className="mt-2 leading-snug">{item.question}</p>
                </button>
              ))}
            </div>
            <p className="text-xs text-slate-400 mt-4">
              Analyst questions from recent EMS earnings calls
            </p>
          </div>
        ) : (
          <div className="space-y-6 max-w-4xl mx-auto">
            {messages.map((msg) => (
              <div key={msg.id} className={`flex gap-3 ${msg.role === 'user' ? 'justify-end' : ''}`}>
                {msg.role === 'assistant' && (
                  <div className="w-8 h-8 bg-blue-100 rounded-lg flex items-center justify-center shrink-0 mt-1">
                    <Bot className="w-4 h-4 text-blue-600" />
                  </div>
                )}
                <div className={`max-w-[80%] ${msg.role === 'user' ? 'order-first' : ''}`}>
                  <div
                    className={`rounded-2xl px-4 py-3 ${
                      msg.role === 'user'
                        ? 'bg-blue-600 text-white'
                        : 'bg-white border border-slate-200'
                    }`}
                  >
                    {msg.role === 'assistant' ? (
                      <div className="prose prose-sm prose-slate max-w-none">
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
                      </div>
                    ) : (
                      <p>{msg.content}</p>
                    )}
                  </div>
                  {/* Sources */}
                  {msg.sources && msg.sources.length > 0 && (
                    <div className="mt-2 flex flex-wrap gap-1.5">
                      {msg.sources.slice(0, 4).map((src, i) => (
                        <Badge key={i} variant="outline" className="text-xs font-normal">
                          {src.company} · {src.filing_type} · {src.fiscal_year}
                        </Badge>
                      ))}
                      {msg.mode && (
                        <Badge className={`text-xs ${modeConfig[msg.mode].color}`}>
                          {modeConfig[msg.mode].label}
                        </Badge>
                      )}
                    </div>
                  )}
                </div>
                {msg.role === 'user' && (
                  <div className="w-8 h-8 bg-slate-200 rounded-lg flex items-center justify-center shrink-0 mt-1">
                    <User className="w-4 h-4 text-slate-600" />
                  </div>
                )}
              </div>
            ))}
            {isLoading && (
              <div className="flex gap-3">
                <div className="w-8 h-8 bg-blue-100 rounded-lg flex items-center justify-center shrink-0">
                  <Bot className="w-4 h-4 text-blue-600" />
                </div>
                <div className="bg-white border border-slate-200 rounded-2xl px-4 py-3">
                  <Loader2 className="w-5 h-5 text-blue-500 animate-spin" />
                </div>
              </div>
            )}
            <div ref={scrollRef} />
          </div>
        )}
      </ScrollArea>

      {/* Input Area */}
      <div className="border-t bg-white px-6 py-4">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            sendMessage();
          }}
          className="flex gap-3 max-w-4xl mx-auto"
        >
          <Input
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask about CapEx, financials, competitive analysis..."
            disabled={isLoading}
            className="flex-1"
            autoFocus
          />
          <Button type="submit" disabled={isLoading || !input.trim()}>
            {isLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
          </Button>
        </form>
        <p className="text-xs text-slate-400 text-center mt-2">
          Using {modeConfig[mode].desc} · Powered by Claude + ChromaDB ({mode === 'rag' ? '19K+ document chunks' : mode === 'web' ? 'Brave Search' : 'Documents + Web'})
        </p>
      </div>
    </div>
  );
}
