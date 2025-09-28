/** @type {import('tailwindcss').Config} */
export default {
  // CRITICAL: This content list tells Tailwind to scan all files in the 
  // 'src' directory ending in .js, .ts, .jsx, or .tsx for utility classes.
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}
