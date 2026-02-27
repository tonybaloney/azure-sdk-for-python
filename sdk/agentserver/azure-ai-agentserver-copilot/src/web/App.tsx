import './App.css'
import { ChatWindow } from './components/ChatWindow'
import { MessageInput } from './components/MessageInput'
import { ThemeToggle } from './components/ThemeToggle'
import { useChat } from './hooks/useChat'
import { useTheme } from './hooks/useTheme'

export default function App() {
  const { messages, isLoading, sendMessage } = useChat()
  const { theme, toggleTheme } = useTheme()

  return (
    <>
      <header className="app-header">
        <div>
          <h1>Copilot Agent Chat</h1>
          <p>Chat with the Copilot SDK Agent. Try asking a question.</p>
        </div>
        <ThemeToggle theme={theme} onToggle={toggleTheme} />
      </header>
      <div className="chat-container">
        <ChatWindow messages={messages} isStreaming={isLoading} />
        <MessageInput onSend={sendMessage} disabled={isLoading} />
      </div>
      <footer className="footer">
        Powered by{' '}
        <a href="https://github.com/github/copilot-sdk" target="_blank" rel="noopener noreferrer">
          Copilot SDK
        </a>
        {' · '}
        <a href="https://learn.microsoft.com/azure/ai-foundry/agents/concepts/hosted-agents" target="_blank" rel="noopener noreferrer">
          Azure AI Agent Server
        </a>
      </footer>
    </>
  )
}
