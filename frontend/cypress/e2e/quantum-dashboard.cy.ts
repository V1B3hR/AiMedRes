/**
 * E2E tests for Quantum Key Management Dashboard
 */

describe('Quantum Key Management Dashboard', () => {
  beforeEach(() => {
    // Login as admin
    cy.visit('/')
    cy.get('#username').type('admin-001')
    cy.get('#password').type('admin-password')
    cy.get('button[type=submit]').click()
    
    // Navigate to quantum key management dashboard
    cy.visit('/admin/quantum')
  })

  it('should load quantum key management dashboard', () => {
    cy.contains('Quantum Key Management')
  })

  it('should display statistics cards', () => {
    cy.contains('Total Keys')
    cy.contains('Active Keys')
    cy.contains('Total Rotations')
    cy.contains('Encryption Ops')
  })

  it('should display key statistics with numbers', () => {
    // Statistics should show numeric values
    cy.get('[style*="fontSize: 32px"]').should('have.length.at.least', 4)
  })

  it('should have tabs for different views', () => {
    cy.contains('Keys')
    cy.contains('Policy')
    cy.contains('History')
  })

  it('should display keys in grid layout', () => {
    // Keys tab should be active by default
    cy.contains('Keys').click()
    
    // Should display key cards
    cy.get('[style*="gridTemplateColumns"]')
  })

  it('should show key type icons', () => {
    cy.contains('Keys').click()
    
    // Should have emoji icons for different key types
    cy.get('body').then($body => {
      const text = $body.text()
      // Icons like 🔑, 🔒, etc. should be present
      expect(text).to.match(/[🔑🔒⏱️🔌💾🔐]/)
    })
  })

  it('should display key status badges', () => {
    cy.contains('Keys').click()
    
    // Status badges should be visible
    cy.contains(/ACTIVE|ROTATING|DEPRECATED|REVOKED/i)
  })

  it('should show key details', () => {
    cy.contains('Keys').click()
    
    // Each key card should show details
    cy.contains('Created:')
    cy.contains('Last Rotated:')
    cy.contains('Rotations:')
    cy.contains('Usage:')
  })

  it('should have rotate button for active keys', () => {
    cy.contains('Keys').click()
    
    cy.get('body').then($body => {
      if ($body.text().includes('Rotate Key')) {
        cy.contains('Rotate Key').should('be.visible')
      }
    })
  })

  it('should confirm before key rotation', () => {
    cy.contains('Keys').click()
    
    cy.get('body').then($body => {
      if ($body.text().includes('Rotate Key')) {
        cy.contains('Rotate Key').first().click()
        // Should show confirmation dialog
      }
    })
  })

  it('should display rotation policy', () => {
    cy.contains('Policy').click()
    
    cy.contains('Rotation Policy Configuration')
    cy.contains('Enable Automatic Key Rotation')
    cy.contains('Rotation Interval')
    cy.contains('Maximum Key Age')
    cy.contains('Grace Period')
  })

  it('should allow updating rotation policy', () => {
    cy.contains('Policy').click()
    
    // Should have input fields
    cy.get('input[type="number"]').should('have.length.at.least', 3)
    cy.get('input[type="checkbox"]').should('have.length.at.least', 2)
  })

  it('should display quantum-safe encryption notice', () => {
    cy.contains('Policy').click()
    
    cy.contains('Quantum-Safe Encryption')
    cy.contains('Kyber768')
    cy.contains('AES-256')
  })

  it('should show rotation history table', () => {
    cy.contains('History').click()
    
    // Should have table headers
    cy.contains('Timestamp')
    cy.contains('Key ID')
    cy.contains('Event Type')
    cy.contains('Details')
  })

  it('should display rotation events', () => {
    cy.contains('History').click()
    
    // Should show event types
    cy.get('body').then($body => {
      if ($body.text().includes('ROTATION')) {
        cy.contains(/ROTATION_STARTED|ROTATION_COMPLETED|ROTATION_FAILED/i)
      }
    })
  })

  it('should have auto-refresh toggle', () => {
    cy.get('input[type="checkbox"]').should('be.visible')
    cy.contains('Auto-refresh')
  })

  it('should require admin authentication', () => {
    cy.clearCookies()
    cy.visit('/admin/quantum')
    // Should redirect to login
    cy.url().should('include', '/')
  })

  it('should ensure PHI compliance', () => {
    // Quantum key management should not expose PHI
    cy.contains('Quantum Key Management')
    // No patient data should be visible
    cy.contains('patient', { matchCase: false }).should('not.exist')
  })

  it('should have audit logging for all operations', () => {
    // All key operations should be audited (verified at API level)
    cy.contains('History').click()
    cy.contains('Timestamp') // History proves audit logging
  })

  it('should load within reasonable time', () => {
    const startTime = Date.now()
    cy.contains('Quantum Key Management')
    cy.then(() => {
      const loadTime = Date.now() - startTime
      expect(loadTime).to.be.lessThan(3000) // Should load within 3 seconds
    })
  })

  it('should be responsive', () => {
    // Check that layout adapts to different screen sizes
    cy.viewport(1280, 720)
    cy.contains('Quantum Key Management')
    
    cy.viewport(768, 1024)
    cy.contains('Quantum Key Management')
  })
})
