type TitleCardProps = {
  eyebrow: string;
  title: string;
  description?: string;
};

export function TitleCard({ eyebrow, title, description }: TitleCardProps) {
  return (
    <section className="title-card">
      <span className="eyebrow">{eyebrow}</span>
      <h1>{title}</h1>
      {description ? <p>{description}</p> : null}
    </section>
  );
}
