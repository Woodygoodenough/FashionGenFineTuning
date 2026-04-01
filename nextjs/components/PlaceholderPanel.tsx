type PlaceholderPanelProps = {
  text: string;
};

export function PlaceholderPanel({ text }: PlaceholderPanelProps) {
  return (
    <section className="placeholder-panel">
      <p>{text}</p>
    </section>
  );
}
