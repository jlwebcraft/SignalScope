import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SignalScope — Image Forensics & Authenticity Verification",
  description:
    "Independent media forensic verification using spatial feature attribution, 2D Fourier spectral harmonics, and perturbation stability.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-[#fbfbf9] text-[#2a2d34] min-h-screen antialiased selection:bg-[#9a3412]/15 selection:text-[#121316]">
        {children}
      </body>
    </html>
  );
}
