import type { Metadata } from "next";
import { JetBrains_Mono } from "next/font/google";
import "./globals.css";

const jetbrainsMono = JetBrains_Mono({
  variable: "--font-jetbrains-mono",
  subsets: ["latin"],
  weight: ["400", "500", "700"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "AeroPINN — Physics-Guided Neural Surrogates for Aerodynamic Prediction",
  description:
    "Investigating whether aerodynamic physics can improve neural-network generalization across unseen airfoil geometries and flight conditions.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`${jetbrainsMono.variable} h-full antialiased`} data-theme="dark">
      <body className="min-h-full flex flex-col bg-bg text-text">{children}</body>
    </html>
  );
}
