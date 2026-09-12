import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SignalScope — Multimodal Authenticity Intelligence",
  description: "Evidence-grounded image authenticity and synthetic media detection system with spatial attribution, 2D spectral cues, and robustness stability.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="bg-[#090d16] text-slate-100 min-h-screen antialiased">
        {children}
      </body>
    </html>
  );
}
