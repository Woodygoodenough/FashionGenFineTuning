"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const navItems = [
  { href: "/", label: "Production Demo" },
  { href: "/embedding-analysis", label: "Embedding Analysis" },
  { href: "/model-evaluation", label: "Model Evaluation" },
  { href: "/ann-search-efficiency", label: "ANN Search Efficiency" },
];

export function SideNav() {
  const pathname = usePathname();

  return (
    <aside className="side-nav">
      <div className="side-nav-brand">
        <span className="eyebrow">FashionGen</span>
        <strong>Interface</strong>
      </div>

      <nav className="side-nav-links" aria-label="Primary">
        {navItems.map((item) => {
          const active = pathname === item.href;

          return (
            <Link
              key={item.href}
              href={item.href}
              className={`side-nav-link${active ? " is-active" : ""}`}
            >
              {item.label}
            </Link>
          );
        })}
      </nav>
    </aside>
  );
}
