/**
 * DICOM Viewer Component
 * Uses Cornerstone.js for medical image viewing
 */

import React, { useRef, useEffect, useState } from 'react'
import * as cornerstone from 'cornerstone-core'
import * as cornerstoneWADOImageLoader from 'cornerstone-wado-image-loader'
import * as dicomParser from 'dicom-parser'

// Initialize Cornerstone WADO Image Loader
cornerstoneWADOImageLoader.external.cornerstone = cornerstone
cornerstoneWADOImageLoader.external.dicomParser = dicomParser

interface DICOMViewerProps {
  imageId?: string
  studyId?: string
  seriesId?: string
  onImageLoad?: (image: any) => void
  onError?: (error: Error) => void
}

export default function DICOMViewer({
  imageId,
  studyId,
  seriesId,
  onImageLoad,
  onError,
}: DICOMViewerProps) {
  const viewportRef = useRef<HTMLDivElement>(null)
  const [viewport, setViewport] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [windowWidth, setWindowWidth] = useState(400)
  const [windowCenter, setWindowCenter] = useState(40)

  useEffect(() => {
    if (!viewportRef.current) return

    // Enable the viewport element
    cornerstone.enable(viewportRef.current)
    
    const vp = cornerstone.getEnabledElement(viewportRef.current)
    setViewport(vp)

    return () => {
      if (viewportRef.current) {
        try {
          cornerstone.disable(viewportRef.current)
        } catch (e) {
          // Element may already be disabled
        }
      }
    }
  }, [])

  useEffect(() => {
    if (!imageId || !viewport) return

    setLoading(true)

    cornerstone
      .loadImage(imageId)
      .then((image) => {
        cornerstone.displayImage(viewportRef.current!, image)
        setLoading(false)
        onImageLoad?.(image)
      })
      .catch((error) => {
        setLoading(false)
        onError?.(error)
        console.error('Failed to load DICOM image:', error)
      })
  }, [imageId, viewport])

  const adjustWindow = (width: number, center: number) => {
    setWindowWidth(width)
    setWindowCenter(center)
    
    if (viewport && viewportRef.current) {
      const vp = cornerstone.getViewport(viewportRef.current)
      vp.voi.windowWidth = width
      vp.voi.windowCenter = center
      cornerstone.setViewport(viewportRef.current, vp)
    }
  }

  const resetView = () => {
    if (viewport && viewportRef.current) {
      cornerstone.reset(viewportRef.current)
    }
  }

  const zoom = (factor: number) => {
    if (viewport && viewportRef.current) {
      const vp = cornerstone.getViewport(viewportRef.current)
      vp.scale *= factor
      cornerstone.setViewport(viewportRef.current, vp)
    }
  }

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      {/* Viewport */}
      <div
        ref={viewportRef}
        style={{
          width: '100%',
          height: 'calc(100% - 60px)',
          backgroundColor: '#000',
        }}
      />
      
      {/* Loading indicator */}
      {loading && (
        <div style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          color: 'white',
          fontSize: '18px',
        }}>
          Loading DICOM image...
        </div>
      )}
      
      {/* Controls */}
      <div style={{
        height: '60px',
        display: 'flex',
        alignItems: 'center',
        gap: '10px',
        padding: '0 15px',
        backgroundColor: '#2a2a2a',
        color: 'white',
      }}>
        <button
          onClick={() => zoom(1.2)}
          style={{ padding: '8px 16px', cursor: 'pointer' }}
        >
          Zoom In
        </button>
        <button
          onClick={() => zoom(0.8)}
          style={{ padding: '8px 16px', cursor: 'pointer' }}
        >
          Zoom Out
        </button>
        <button
          onClick={resetView}
          style={{ padding: '8px 16px', cursor: 'pointer' }}
        >
          Reset
        </button>
        
        <div style={{ marginLeft: 'auto', display: 'flex', gap: '15px', fontSize: '14px' }}>
          <div>
            <label>W: </label>
            <input
              type="range"
              min="1"
              max="2000"
              value={windowWidth}
              onChange={(e) => adjustWindow(Number(e.target.value), windowCenter)}
              style={{ width: '100px' }}
            />
            <span style={{ marginLeft: '5px' }}>{windowWidth}</span>
          </div>
          <div>
            <label>C: </label>
            <input
              type="range"
              min="-1000"
              max="1000"
              value={windowCenter}
              onChange={(e) => adjustWindow(windowWidth, Number(e.target.value))}
              style={{ width: '100px' }}
            />
            <span style={{ marginLeft: '5px' }}>{windowCenter}</span>
          </div>
        </div>
      </div>
    </div>
  )
}
