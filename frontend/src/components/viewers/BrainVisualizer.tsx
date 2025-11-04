/**
 * 3D Brain Visualization Component
 * Uses Three.js and React Three Fiber for interactive 3D brain rendering
 */

import React, { useRef, useState, useEffect, Suspense } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, PerspectiveCamera, Html } from '@react-three/drei'
import * as THREE from 'three'

interface BrainRegion {
  region_id: string
  name: string
  coordinates: number[]
  volume: number
  affected: boolean
  severity?: number
}

interface BrainVisualizerProps {
  regions: BrainRegion[]
  diseaseStage?: string
  explainability?: {
    affected_regions: string[]
    confidence: number
  }
  onRegionClick?: (region: BrainRegion) => void
}

function BrainMesh({ regions, onRegionClick }: { regions: BrainRegion[], onRegionClick?: (region: BrainRegion) => void }) {
  const meshRef = useRef<THREE.Group>(null)
  const [hoveredRegion, setHoveredRegion] = useState<string | null>(null)

  useFrame(() => {
    if (meshRef.current) {
      // Subtle rotation for better visibility
      meshRef.current.rotation.y += 0.001
    }
  })

  return (
    <group ref={meshRef}>
      {regions.map((region, index) => {
        const [x, y, z] = region.coordinates
        const color = region.affected
          ? region.severity && region.severity > 0.7
            ? '#ff0000'
            : '#ff8800'
          : '#88ccff'
        const scale = region.affected ? 1.2 : 1.0

        return (
          <mesh
            key={region.region_id}
            position={[x, y, z]}
            scale={scale}
            onClick={() => onRegionClick?.(region)}
            onPointerOver={() => setHoveredRegion(region.region_id)}
            onPointerOut={() => setHoveredRegion(null)}
          >
            <sphereGeometry args={[0.3, 32, 32]} />
            <meshStandardMaterial
              color={color}
              transparent
              opacity={hoveredRegion === region.region_id ? 0.9 : 0.7}
              emissive={region.affected ? '#ff4400' : '#004488'}
              emissiveIntensity={region.affected ? 0.3 : 0.1}
            />
            {hoveredRegion === region.region_id && (
              <Html distanceFactor={10}>
                <div style={{
                  background: 'rgba(0, 0, 0, 0.8)',
                  color: 'white',
                  padding: '8px',
                  borderRadius: '4px',
                  fontSize: '12px',
                  whiteSpace: 'nowrap',
                }}>
                  <div><strong>{region.name}</strong></div>
                  <div>Volume: {region.volume.toFixed(2)}</div>
                  {region.affected && (
                    <div>Severity: {((region.severity || 0) * 100).toFixed(0)}%</div>
                  )}
                </div>
              </Html>
            )}
          </mesh>
        )
      })}
    </group>
  )
}

export default function BrainVisualizer({
  regions,
  diseaseStage,
  explainability,
  onRegionClick,
}: BrainVisualizerProps) {
  return (
    <div style={{ width: '100%', height: '600px', position: 'relative' }}>
      <Canvas>
        <Suspense fallback={null}>
          <PerspectiveCamera makeDefault position={[0, 0, 10]} />
          <OrbitControls enableDamping dampingFactor={0.05} />
          
          {/* Lighting */}
          <ambientLight intensity={0.5} />
          <directionalLight position={[10, 10, 5]} intensity={1} />
          <directionalLight position={[-10, -10, -5]} intensity={0.5} />
          
          {/* Brain regions */}
          <BrainMesh regions={regions} onRegionClick={onRegionClick} />
          
          {/* Grid helper for reference */}
          <gridHelper args={[10, 10]} />
        </Suspense>
      </Canvas>
      
      {/* Info panel */}
      <div style={{
        position: 'absolute',
        top: '10px',
        left: '10px',
        background: 'rgba(0, 0, 0, 0.7)',
        color: 'white',
        padding: '15px',
        borderRadius: '8px',
        maxWidth: '300px',
      }}>
        <h3 style={{ margin: '0 0 10px 0', fontSize: '16px' }}>
          3D Brain Visualization
        </h3>
        {diseaseStage && (
          <div style={{ marginBottom: '8px' }}>
            <strong>Disease Stage:</strong> {diseaseStage}
          </div>
        )}
        <div style={{ marginBottom: '8px' }}>
          <strong>Regions:</strong> {regions.length}
        </div>
        <div style={{ marginBottom: '8px' }}>
          <strong>Affected:</strong> {regions.filter(r => r.affected).length}
        </div>
        {explainability && (
          <>
            <div style={{ marginBottom: '8px' }}>
              <strong>AI Confidence:</strong> {(explainability.confidence * 100).toFixed(1)}%
            </div>
            <div style={{ fontSize: '11px', color: '#aaa' }}>
              Click regions for details
            </div>
          </>
        )}
      </div>
    </div>
  )
}
