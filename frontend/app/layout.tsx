import './globals.css';

export const metadata = {
  title: 'AYU Health',
  description: 'AI-powered medical report analysis prototype',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
