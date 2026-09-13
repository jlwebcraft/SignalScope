import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SignalScope — Digital Media Forensic Workbench",
  description:
    "Professional image authenticity forensics with spatial attribution, 2D Fourier spectral analysis, and perturbation robustness probing.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="bg-[#0b0f17] text-slate-100 min-h-screen antialiased">
        {children}
      </body>
    </html>
  );
}
