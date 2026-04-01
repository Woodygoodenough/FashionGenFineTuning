import "./globals.css";
import type { Metadata } from "next";
import { SideNav } from "../components/SideNav";

export const metadata: Metadata = {
  title: "Fashion Retrieval Demo",
  description: "Deployment-first scaffold for multimodal fashion retrieval"
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div className="app-shell">
          <SideNav />
          <div className="app-main">{children}</div>
        </div>
      </body>
    </html>
  );
}
