import { useEffect, useRef, useState, useCallback, lazy, Suspense } from 'react'
import ReactMarkdown from 'react-markdown'
import type { Message } from '../types'

const LazyHighlighter = lazy(() =>
  Promise.all([
    import('react-syntax-highlighter/dist/esm/prism-light'),
    import('react-syntax-highlighter/dist/esm/styles/prism/one-dark'),
  ]).then(([{ default: SyntaxHighlighter }, { default: oneDark }]) => ({
    default: ({ language, code }: { language: string; code: string }) => (
      <SyntaxHighlighter style={oneDark} language={language} PreTag="div">
        {code}
      </SyntaxHighlighter>
    ),
  }))
)

interface Props {
  messages: Message[]
  isStreaming: boolean
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false)

  const handleCopy = useCallback(() => {
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    }).catch(err => console.error('Failed to copy:', err))
  }, [text])

  return (
    <button className="copy-btn" onClick={handleCopy} title="Copy code" aria-label="Copy code">
      {copied ? (
        <svg viewBox="0 0 16 16" width="14" height="14" fill="currentColor"><path d="M13.78 4.22a.75.75 0 0 1 0 1.06l-7.25 7.25a.75.75 0 0 1-1.06 0L2.22 9.28a.751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018L6 10.94l6.72-6.72a.75.75 0 0 1 1.06 0Z"/></svg>
      ) : (
        <svg viewBox="0 0 16 16" width="14" height="14" fill="currentColor"><path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 0 1 0 1.5h-1.5a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-1.5a.75.75 0 0 1 1.5 0v1.5A1.75 1.75 0 0 1 9.25 16h-7.5A1.75 1.75 0 0 1 0 14.25ZM5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0 1 14.25 11h-7.5A1.75 1.75 0 0 1 5 9.25Zm1.75-.25a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-7.5a.25.25 0 0 0-.25-.25Z"/></svg>
      )}
    </button>
  )
}

function CopilotLogo() {
  return (
    <svg className="loading-logo" viewBox="0 0 16 16" width="24" height="24" aria-hidden="true">
      <path fill="currentColor" d="M8 0c4.42 0 8 3.58 8 8a8.013 8.013 0 0 1-5.45 7.59c-.4.08-.55-.17-.55-.38 0-.27.01-1.13.01-2.2 0-.75-.25-1.23-.54-1.48 1.78-.2 3.65-.88 3.65-3.95 0-.88-.31-1.59-.82-2.15.08-.2.36-1.02-.08-2.12 0 0-.67-.22-2.2.82-.64-.18-1.32-.27-2-.27-.68 0-1.36.09-2 .27-1.53-1.03-2.2-.82-2.2-.82-.44 1.1-.16 1.92-.08 2.12-.51.56-.82 1.28-.82 2.15 0 3.06 1.86 3.75 3.64 3.95-.23.2-.44.55-.51 1.07-.46.21-1.61.55-2.33-.66-.15-.24-.6-.83-1.23-.82-.67.01-.27.38.01.53.34.19.73.9.82 1.13.16.45.68 1.31 2.69.94 0 .67.01 1.3.01 1.49 0 .21-.15.45-.55.38A7.995 7.995 0 0 1 0 8c0-4.42 3.58-8 8-8Z"/>
    </svg>
  )
}

export function ChatWindow({ messages, isStreaming }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  if (messages.length === 0) {
    return (
      <div className="messages" role="log" aria-live="polite">
        <div className="empty-state">Send a message to get started</div>
      </div>
    )
  }

  return (
    <div className="messages" role="log" aria-live="polite">
      {messages.map((msg, i) => {
        const isLast = i === messages.length - 1
        const streaming = msg.role === 'assistant' && isStreaming && isLast
        const showLoader = streaming && !msg.content

        return (
          <div key={msg.id} className={`message ${msg.role}${streaming ? ' streaming' : ''}`}>
            {showLoader ? (
              <CopilotLogo />
            ) : msg.role === 'assistant' ? (
              <ReactMarkdown
                components={{
                  code({ className, children, ...props }) {
                    const match = /language-(\w+)/.exec(className || '')
                    const code = String(children).replace(/\n$/, '')
                    if (match) {
                      return (
                        <div className="code-block-wrapper">
                          <CopyButton text={code} />
                          <Suspense fallback={<pre><code>{code}</code></pre>}>
                            <LazyHighlighter language={match[1]} code={code} />
                          </Suspense>
                        </div>
                      )
                    }
                    return <code className={className} {...props}>{children}</code>
                  },
                }}
              >
                {msg.content}
              </ReactMarkdown>
            ) : (
              msg.content
            )}
          </div>
        )
      })}
      <div ref={bottomRef} />
    </div>
  )
}
