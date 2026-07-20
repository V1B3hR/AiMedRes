const implementedFeatures = [
  'Clinical case APIs with Vitest coverage',
  'Canary deployment monitoring dashboard modules',
  'Quantum key management dashboard modules',
  '3D brain visualization and DICOM viewer components',
]

export default function App() {
  return (
    <main
      style={{
        fontFamily: 'Arial, sans-serif',
        margin: '0 auto',
        maxWidth: '960px',
        padding: '3rem 1.5rem',
      }}
    >
      <h1>AiMedRes Frontend</h1>
      <p>
        Research UI assets for clinical decision support, visualization, deployment
        monitoring, and security operations.
      </p>

      <section>
        <h2>Implemented modules</h2>
        <ul>
          {implementedFeatures.map((feature) => (
            <li key={feature}>{feature}</li>
          ))}
        </ul>
      </section>
    </main>
  )
}
