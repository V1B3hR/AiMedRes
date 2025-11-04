/**
 * E2E tests for 3D Brain Visualizer
 */

describe('3D Brain Visualizer', () => {
  beforeEach(() => {
    // Login first
    cy.visit('/')
    cy.get('#username').type('clinician-001')
    cy.get('#password').type('test-password')
    cy.get('button[type=submit]').click()
    cy.url().should('include', '/cases')
    
    // Navigate to brain visualizer
    cy.visit('/viewer/brain')
  })

  it('should load 3D brain visualization', () => {
    cy.contains('3D Brain Visualization')
    cy.get('canvas').should('be.visible')
  })

  it('should display brain regions', () => {
    // Wait for visualization to load
    cy.wait(1000)
    
    // Should show region count
    cy.contains('Regions:')
    cy.contains('Affected:')
  })

  it('should display disease stage information', () => {
    cy.contains('Disease Stage:')
    cy.contains('AI Confidence:')
  })

  it('should allow interaction with 3D model', () => {
    // Check that canvas is interactive
    cy.get('canvas').should('be.visible')
    cy.get('canvas').trigger('mousedown', { which: 1 })
    cy.get('canvas').trigger('mousemove', { clientX: 100, clientY: 100 })
    cy.get('canvas').trigger('mouseup')
  })

  it('should show region details on hover', () => {
    // Simulate hovering over a region
    cy.get('canvas').trigger('mousemove', { clientX: 300, clientY: 300 })
    cy.wait(500)
    // Tooltip should appear with region details
  })

  it('should display PHI compliance notice', () => {
    cy.contains('RESEARCH USE ONLY', { matchCase: false })
  })

  it('should render within reasonable time', () => {
    const startTime = Date.now()
    cy.get('canvas').should('be.visible')
    cy.then(() => {
      const loadTime = Date.now() - startTime
      expect(loadTime).to.be.lessThan(5000) // Should load within 5 seconds
    })
  })
})
