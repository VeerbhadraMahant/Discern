export default function AnnotatedFrame({ src }: { src: string }) {
  return (
    <div className="overflow-hidden rounded-card-lg bg-obsidian shadow-feature">
      <img src={src} alt="Analyzed camera frame with detection overlays" className="block w-full" />
    </div>
  );
}
