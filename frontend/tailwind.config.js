/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  darkMode: ["class"],
  theme: {
    extend: {
      colors: {
        obsidian: "#070709",
        paper: "#ffffff",
        cloud: "#f5f5f5",
        ink: "#151515",
        charcoal: "#60606c",
        slate: "#8b8b8b",
        "sky-tint": "#d7e6f5",
        accent: "#2597d0",
        critical: "#c8362a",
        warning: "#b8720b",
        safe: "#3a8a52",
      },
      fontFamily: {
        heading: ["Outfit", "system-ui", "sans-serif"],
        body: ["Inter", "system-ui", "sans-serif"],
      },
      borderRadius: {
        pill: "100px",
        card: "18px",
        "card-lg": "32px",
        input: "12px",
        icon: "8px",
      },
      boxShadow: {
        feature: "0 17px 37px rgba(16,55,132,0.03), 0 67px 67px rgba(16,55,132,0.03), 0 150px 90px rgba(16,55,132,0.02)",
        pill: "0 1px 2px rgba(36,36,40,0.1), 0 3px 3px rgba(36,36,40,0.09), 0 6px 4px rgba(36,36,40,0.05)",
        subtle: "0 1px 2px rgba(228,229,231,0.24)",
      },
      backgroundImage: {
        "sky-gradient": "linear-gradient(180deg, #779bc1 0%, #9abfda 58%, #cbdcec 100%)",
      },
    },
  },
  plugins: [],
};
