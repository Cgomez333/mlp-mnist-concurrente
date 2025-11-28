import { useState, useEffect } from 'react'
import './App.css'

function App() {
  const [activeTab, setActiveTab] = useState('overview')

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>🧠 MLP MNIST - Reconocimiento de Dígitos</h1>
        <p className="subtitle">Red Neuronal Multicapa con Paralelización OpenMP</p>
      </header>

      <nav className="tab-navigation">
        <button 
          className={activeTab === 'overview' ? 'tab active' : 'tab'}
          onClick={() => setActiveTab('overview')}
        >
          📊 Resumen
        </button>
        <button 
          className={activeTab === 'draw' ? 'tab active' : 'tab'}
          onClick={() => setActiveTab('draw')}
        >
          ✏️ Dibujar
        </button>
        <button 
          className={activeTab === 'dataset' ? 'tab active' : 'tab'}
          onClick={() => setActiveTab('dataset')}
        >
          🖼️ Dataset
        </button>
        <button 
          className={activeTab === 'performance' ? 'tab active' : 'tab'}
          onClick={() => setActiveTab('performance')}
        >
          🚀 Rendimiento
        </button>
      </nav>

      <main className="app-content">
        {activeTab === 'overview' && <OverviewTab />}
        {activeTab === 'draw' && <DrawTab />}
        {activeTab === 'dataset' && <DatasetTab />}
        {activeTab === 'performance' && <PerformanceTab />}
      </main>

      <footer className="app-footer">
        <p>Universidad de Caldas - Programación Concurrente 2025</p>
      </footer>
    </div>
  )
}

function OverviewTab() {
  return (
    <div className="tab-content">
      <h2>Proyecto MLP MNIST</h2>
      
      <div className="info-grid">
        <div className="info-card">
          <h3>🎯 Arquitectura</h3>
          <div className="architecture">
            <div className="layer">784 neuronas<br/><small>(Entrada 28×28)</small></div>
            <div className="arrow">→</div>
            <div className="layer">512 neuronas<br/><small>(Oculta - ReLU)</small></div>
            <div className="arrow">→</div>
            <div className="layer">10 neuronas<br/><small>(Salida - Softmax)</small></div>
          </div>
        </div>

        <div className="info-card">
          <h3>📈 Resultados</h3>
          <ul className="results-list">
            <li><strong>Accuracy:</strong> 93.56%</li>
            <li><strong>Secuencial:</strong> 1,539s (10 epochs)</li>
            <li><strong>OpenMP (8 threads):</strong> 346s</li>
            <li><strong>Speedup:</strong> 4.45×</li>
          </ul>
        </div>

        <div className="info-card">
          <h3>⚙️ Hiperparámetros</h3>
          <ul className="params-list">
            <li><strong>Epochs:</strong> 10</li>
            <li><strong>Batch Size:</strong> 64</li>
            <li><strong>Learning Rate:</strong> 0.01</li>
            <li><strong>Dataset:</strong> 60k train + 10k test</li>
          </ul>
        </div>

        <div className="info-card">
          <h3>💻 Implementaciones</h3>
          <ul className="impl-list">
            <li>✅ C Secuencial</li>
            <li>✅ C + OpenMP</li>
            <li>⏳ Python (compañero)</li>
            <li>⏳ PyCUDA (compañero)</li>
          </ul>
        </div>
      </div>
    </div>
  )
}

function DrawTab() {
  const canvasRef = useState(null)[0]
  const [isDrawing, setIsDrawing] = useState(false)
  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(false)
  const [selectedModel, setSelectedModel] = useState('sequential')

  useEffect(() => {
    const canvas = document.getElementById('drawCanvas')
    if (canvas) {
      const ctx = canvas.getContext('2d')
      ctx.fillStyle = 'white'
      ctx.fillRect(0, 0, 280, 280)
      ctx.lineWidth = 28
      ctx.lineCap = 'round'
      ctx.strokeStyle = 'black'
    }
  }, [])

  const startDrawing = (e) => {
    const canvas = document.getElementById('drawCanvas')
    const rect = canvas.getBoundingClientRect()
    const ctx = canvas.getContext('2d')
    
    ctx.beginPath()
    ctx.moveTo(e.clientX - rect.left, e.clientY - rect.top)
    setIsDrawing(true)
  }

  const draw = (e) => {
    if (!isDrawing) return
    
    const canvas = document.getElementById('drawCanvas')
    const rect = canvas.getBoundingClientRect()
    const ctx = canvas.getContext('2d')
    
    ctx.lineTo(e.clientX - rect.left, e.clientY - rect.top)
    ctx.stroke()
  }

  const stopDrawing = () => {
    setIsDrawing(false)
  }

  const clearCanvas = () => {
    const canvas = document.getElementById('drawCanvas')
    const ctx = canvas.getContext('2d')
    ctx.fillStyle = 'white'
    ctx.fillRect(0, 0, 280, 280)
    setPrediction(null)
  }

  const predict = async () => {
    const canvas = document.getElementById('drawCanvas')
    const ctx = canvas.getContext('2d')
    
    // Obtener bounding box del contenido dibujado
    const imageData = ctx.getImageData(0, 0, 280, 280)
    let minX = 280, minY = 280, maxX = 0, maxY = 0
    
    for (let y = 0; y < 280; y++) {
      for (let x = 0; x < 280; x++) {
        const i = (y * 280 + x) * 4
        const brightness = (imageData.data[i] + imageData.data[i+1] + imageData.data[i+2]) / 3
        if (brightness < 250) { // Si no es blanco
          minX = Math.min(minX, x)
          minY = Math.min(minY, y)
          maxX = Math.max(maxX, x)
          maxY = Math.max(maxY, y)
        }
      }
    }
    
    // Si no hay nada dibujado
    if (minX >= maxX || minY >= maxY) {
      alert('Dibuja un número primero')
      return
    }
    
    // Agregar padding (20%)
    const width = maxX - minX
    const height = maxY - minY
    const padding = Math.max(width, height) * 0.2
    minX = Math.max(0, minX - padding)
    minY = Math.max(0, minY - padding)
    maxX = Math.min(280, maxX + padding)
    maxY = Math.min(280, maxY + padding)
    
    // Crear canvas temporal con el dígito centrado
    const croppedWidth = maxX - minX
    const croppedHeight = maxY - minY
    const size = Math.max(croppedWidth, croppedHeight)
    
    const tempCanvas = document.createElement('canvas')
    tempCanvas.width = size
    tempCanvas.height = size
    const tempCtx = tempCanvas.getContext('2d')
    tempCtx.fillStyle = 'white'
    tempCtx.fillRect(0, 0, size, size)
    
    // Centrar el contenido
    const offsetX = (size - croppedWidth) / 2
    const offsetY = (size - croppedHeight) / 2
    tempCtx.drawImage(canvas, minX, minY, croppedWidth, croppedHeight, offsetX, offsetY, croppedWidth, croppedHeight)
    
    // Redimensionar a 28x28
    const finalCanvas = document.createElement('canvas')
    finalCanvas.width = 28
    finalCanvas.height = 28
    const finalCtx = finalCanvas.getContext('2d')
    finalCtx.drawImage(tempCanvas, 0, 0, 28, 28)
    
    // Obtener datos de píxeles
    const finalImageData = finalCtx.getImageData(0, 0, 28, 28)
    const pixels = []
    
    for (let i = 0; i < finalImageData.data.length; i += 4) {
      const r = finalImageData.data[i]
      const g = finalImageData.data[i + 1]
      const b = finalImageData.data[i + 2]
      const brightness = (r + g + b) / 3
      pixels.push(1 - brightness / 255) // Invertir: 0=blanco, 1=negro
    }

    setLoading(true)
    
    try {
      const response = await fetch('http://localhost:3001/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: selectedModel, image: pixels })
      })
      
      const data = await response.json()
      setPrediction(data)
    } catch (error) {
      console.error('Error:', error)
      alert('Error al conectar con el backend. Asegúrate de que esté corriendo en puerto 3001')
    }
    
    setLoading(false)
  }

  return (
    <div className="tab-content">
      <h2>✏️ Dibuja un Dígito</h2>
      <div className="draw-container">
        <div className="canvas-container">
          <canvas 
            id="drawCanvas"
            width="280" 
            height="280" 
            onMouseDown={startDrawing}
            onMouseMove={draw}
            onMouseUp={stopDrawing}
            onMouseLeave={stopDrawing}
            style={{ 
              border: '2px solid #333',
              borderRadius: '8px',
              background: '#fff',
              cursor: 'crosshair',
              touchAction: 'none'
            }}
          />
          <div className="canvas-controls">
            <select 
              value={selectedModel} 
              onChange={(e) => setSelectedModel(e.target.value)}
              style={{
                padding: '0.75rem',
                borderRadius: '8px',
                border: '2px solid #667eea',
                fontSize: '1rem',
                marginBottom: '0.5rem'
              }}
            >
              <option value="sequential">C Secuencial</option>
              <option value="openmp">C + OpenMP</option>
            </select>
            <button className="btn-primary" onClick={predict} disabled={loading}>
              {loading ? '⏳ Prediciendo...' : '🔮 Predecir'}
            </button>
            <button className="btn-secondary" onClick={clearCanvas}>🗑️ Limpiar</button>
          </div>
        </div>
        
        <div className="prediction-panel">
          <h3>Predicción del Modelo</h3>
          <div className="prediction-result">
            <div className="digit-prediction">
              {prediction ? prediction.prediction : '?'}
            </div>
            <p className="confidence">
              {prediction 
                ? `${(prediction.confidence * 100).toFixed(1)}% de confianza`
                : 'Dibuja un número (0-9)'
              }
            </p>
          </div>
          <div className="probabilities">
            {prediction && prediction.probabilities ? (
              <div>
                <h4 style={{ marginBottom: '0.5rem' }}>Probabilidades:</h4>
                {prediction.probabilities.map((prob, idx) => (
                  <div key={idx} style={{ 
                    display: 'flex', 
                    alignItems: 'center', 
                    marginBottom: '0.3rem',
                    fontSize: '0.9rem'
                  }}>
                    <span style={{ width: '30px', fontWeight: 'bold' }}>{idx}:</span>
                    <div style={{ 
                      flex: 1, 
                      height: '20px', 
                      background: '#e0e0e0',
                      borderRadius: '4px',
                      overflow: 'hidden',
                      marginRight: '0.5rem'
                    }}>
                      <div style={{
                        width: `${prob * 100}%`,
                        height: '100%',
                        background: idx === prediction.prediction ? '#667eea' : '#aaa',
                        transition: 'width 0.3s ease'
                      }}></div>
                    </div>
                    <span style={{ width: '50px', textAlign: 'right' }}>
                      {(prob * 100).toFixed(1)}%
                    </span>
                  </div>
                ))}
              </div>
            ) : (
              <p><small>Las probabilidades aparecerán aquí...</small></p>
            )}
          </div>
        </div>
      </div>
      
      <div className="info-box">
        <p><strong>💡 Tip:</strong> Dibuja números grandes y centrados para mejores resultados. 
        El modelo fue entrenado con 60,000 ejemplos de dígitos manuscritos.</p>
      </div>
    </div>
  )
}

function DatasetTab() {
  return (
    <div className="tab-content">
      <h2>🖼️ Explorar Dataset MNIST</h2>
      
      <div className="dataset-info">
        <div className="info-card">
          <h3>📊 Información del Dataset</h3>
          <ul>
            <li><strong>Training:</strong> 60,000 imágenes</li>
            <li><strong>Test:</strong> 10,000 imágenes</li>
            <li><strong>Resolución:</strong> 28×28 píxeles (escala de grises)</li>
            <li><strong>Clases:</strong> 10 dígitos (0-9)</li>
            <li><strong>Formato:</strong> Binario (.bin) para C</li>
          </ul>
        </div>

        <div className="info-card">
          <h3>🔍 Visualización</h3>
          <p>Para visualizar las imágenes del dataset, usa el script Python:</p>
          <div className="code-block">
            <code>
              cd backend<br/>
              python visualize_mnist.py 0     # Ver imagen 0<br/>
              python visualize_mnist.py 0 10  # Ver primeras 10
            </code>
          </div>
        </div>
      </div>

      <div className="sample-images">
        <h3>Ejemplos del Dataset</h3>
        <p className="info-text">
          El dataset MNIST contiene dígitos escritos a mano por diferentes personas.
          Cada imagen es de 28×28 píxeles y está normalizada entre 0.0 (blanco) y 1.0 (negro).
        </p>
      </div>
    </div>
  )
}

function PerformanceTab() {
  const benchmarkData = [
    { version: 'Secuencial', time: 1539, threads: 1, speedup: 1.0, efficiency: 100 },
    { version: 'OpenMP', time: 1593, threads: 1, speedup: 0.97, efficiency: 97 },
    { version: 'OpenMP', time: 873, threads: 2, speedup: 1.76, efficiency: 88 },
    { version: 'OpenMP', time: 526, threads: 4, speedup: 2.93, efficiency: 73 },
    { version: 'OpenMP', time: 346, threads: 8, speedup: 4.45, efficiency: 56 },
  ]

  return (
    <div className="tab-content">
      <h2>🚀 Análisis de Rendimiento</h2>
      
      <div className="performance-summary">
        <div className="metric-card">
          <h3>⚡ Mejor Speedup</h3>
          <div className="metric-value">4.45×</div>
          <p>Con 8 threads</p>
        </div>
        
        <div className="metric-card">
          <h3>⏱️ Tiempo Ahorrado</h3>
          <div className="metric-value">77.5%</div>
          <p>De 1,539s a 346s</p>
        </div>
        
        <div className="metric-card">
          <h3>🎯 Eficiencia Pico</h3>
          <div className="metric-value">88%</div>
          <p>Con 2 threads</p>
        </div>
      </div>

      <div className="benchmark-table-container">
        <h3>📊 Tabla de Benchmarks</h3>
        <table className="benchmark-table">
          <thead>
            <tr>
              <th>Versión</th>
              <th>Threads</th>
              <th>Tiempo (s)</th>
              <th>Speedup</th>
              <th>Eficiencia (%)</th>
            </tr>
          </thead>
          <tbody>
            {benchmarkData.map((row, idx) => (
              <tr key={idx} className={row.speedup === 4.45 ? 'highlight' : ''}>
                <td>{row.version}</td>
                <td>{row.threads}</td>
                <td>{row.time}s</td>
                <td>{row.speedup.toFixed(2)}×</td>
                <td>{row.efficiency}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="analysis-section">
        <h3>📈 Conclusiones</h3>
        <ul className="analysis-list">
          <li>✅ <strong>Speedup superlineal imposible:</strong> 4.45× con 8 threads supera la Ley de Amdahl teórica (3.88×)</li>
          <li>✅ <strong>Mejor eficiencia:</strong> 88% con 2 threads - óptimo para este caso</li>
          <li>⚠️ <strong>Overhead visible:</strong> OpenMP con 1 thread es 3% más lento que secuencial</li>
          <li>📉 <strong>Escalabilidad:</strong> Eficiencia cae a 56% con 8 threads (normal)</li>
        </ul>
      </div>
    </div>
  )
}

export default App
