/**
 * E2E test for login flow
 */

describe('Login Flow', () => {
  beforeEach(() => {
    cy.visit('/')
  })

  it('should display login page', () => {
    cy.contains('AiMedRes')
    cy.contains('Clinical Decision Support System')
    cy.get('#username').should('be.visible')
    cy.get('#password').should('be.visible')
  })

  it('should show error for empty credentials', () => {
    cy.get('button[type=submit]').click()
    cy.contains('Please enter username and password')
  })

  it('should allow user to login', () => {
    cy.get('#username').type('clinician-001')
    cy.get('#password').type('test-password')
    cy.get('button[type=submit]').click()
    
    // Should redirect to cases page
    cy.url().should('include', '/cases')
    cy.contains('Clinical Cases')
  })

  it('should display research use disclaimer', () => {
    cy.contains('RESEARCH USE ONLY - NOT FOR CLINICAL DIAGNOSIS')
  })
})
