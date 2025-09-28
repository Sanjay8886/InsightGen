import React from 'react';
import ReactDOM from 'react-dom/client';
// FIX: The extension (.tsx) has been removed. TypeScript will now resolve this correctly.
import App from './App'; 
import './index.css'; 

// Get the root element from the HTML
const rootElement = document.getElementById('root');

if (rootElement) {
  ReactDOM.createRoot(rootElement).render(
    <React.StrictMode>
      {/* This is where the component imported above is rendered. */}
      <App />
    </React.StrictMode>,
  );
} else {
    // Log an error if the root element isn't found, preventing a crash.
    console.error("The root element with id 'root' was not found in the DOM.");
}
