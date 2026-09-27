export function Logo({ size = 26 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M8 10 L16 6 L24 10" stroke="#38bdf8" strokeWidth="2" strokeLinecap="round" />
      <path d="M8 10 L8 20 L16 26" stroke="#38bdf8" strokeWidth="2" strokeLinecap="round" />
      <path d="M24 10 L24 20 L16 26" stroke="#38bdf8" strokeWidth="2" strokeLinecap="round" />
      <path d="M16 6 L16 26" stroke="#0ea5e9" strokeWidth="2" strokeLinecap="round" strokeDasharray="1 5" />
      <circle cx="16" cy="6" r="3.5" fill="#0f172a" stroke="#38bdf8" strokeWidth="2" />
      <circle cx="8" cy="10" r="3" fill="#0f172a" stroke="#38bdf8" strokeWidth="2" />
      <circle cx="24" cy="10" r="3" fill="#0f172a" stroke="#38bdf8" strokeWidth="2" />
      <circle cx="16" cy="26" r="3.5" fill="#0f172a" stroke="#34d399" strokeWidth="2" />
    </svg>
  );
}
