import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SignalScope — Image Authenticity & Forensics Workstation",
  description:
    "Specialist image authenticity examination workstation utilizing spatial ConvNeXt feature attribution, 2D Fourier spectral residuals, perturbation stability stress-testing, and C2PA/EXIF provenance.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-[#f8f9fa] text-[#334155] min-h-screen antialiased selection:bg-[#0f172a] selection:text-[#ffffff]">
        {children}
      </body>
    </html>
  );
}
