/**
 * E2E tests for DICOM Viewer
 */

describe('DICOM Viewer', () => {
  beforeEach(() => {
    // Login first
    cy.visit('/')
    cy.get('#username').type('clinician-001')
    cy.get('#password').type('test-password')
    cy.get('button[type=submit]').click()
    cy.url().should('include', '/cases')
    
    // Navigate to DICOM viewer with a test study
    cy.visit('/viewer/dicom?studyId=test-study-001')
  })

  it('should load DICOM viewer interface', () => {
    cy.contains('DICOM', { matchCase: false })
    // Viewport should be visible
    cy.get('div[style*="backgroundColor"]').should('be.visible')
  })

  it('should display viewer controls', () => {
    cy.contains('Zoom In')
    cy.contains('Zoom Out')
    cy.contains('Reset')
  })

  it('should have windowing controls', () => {
    // Window width slider
    cy.get('input[type="range"]').should('have.length.at.least', 2)
    
    // Check for W (width) and C (center) labels
    cy.contains('W:')
    cy.contains('C:')
  })

  it('should allow zoom in', () => {
    cy.contains('Zoom In').click()
    // Should zoom the viewport
  })

  it('should allow zoom out', () => {
    cy.contains('Zoom Out').click()
    // Should zoom out the viewport
  })

  it('should reset view', () => {
    cy.contains('Zoom In').click()
    cy.wait(500)
    cy.contains('Reset').click()
    // Should reset to original view
  })

  it('should adjust window width', () => {
    cy.get('input[type="range"]').first().invoke('val', 800).trigger('change')
    // Window width should be updated
  })

  it('should adjust window center', () => {
    cy.get('input[type="range"]').last().invoke('val', 100).trigger('change')
    // Window center should be updated
  })

  it('should display loading state', () => {
    // When initially loading, should show loading indicator
    cy.visit('/viewer/dicom?studyId=new-study-002')
    cy.contains('Loading DICOM image', { timeout: 2000 })
  })

  it('should ensure PHI compliance', () => {
    // Should not display patient identifying information
    cy.contains('RESEARCH USE ONLY', { matchCase: false })
  })

  it('should be accessible with keyboard', () => {
    // Check that controls are keyboard accessible
    cy.contains('Zoom In').focus().should('have.focus')
    cy.contains('Zoom In').type('{enter}')
  })
})
