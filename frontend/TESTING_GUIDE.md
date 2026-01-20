# Frontend Testing Guide

## Overview

This guide provides comprehensive information about the testing infrastructure for the YouTube Sentiment Analysis frontend application.

## Testing Framework

The application uses the following testing stack:

- **Jest**: JavaScript testing framework
- **React Testing Library**: Testing utilities for React components
- **@testing-library/user-event**: Simulates user interactions
- **@testing-library/jest-dom**: Custom Jest matchers for DOM

## Test Files

### Component Tests

1. **Search.test.js** - Tests for video analysis search functionality
2. **Dashboard.test.js** - Tests for main dashboard display
3. **Monitoring.test.js** - Tests for analysis monitoring and history
4. **Report.test.js** - Tests for report generation and printing

## Running Tests

### Run all tests
```bash
cd frontend
npm test
```

### Run tests in watch mode
```bash
npm test -- --watch
```

### Run tests with coverage
```bash
npm test -- --coverage
```

### Run specific test file
```bash
npm test Search.test.js
```

### Run tests matching a pattern
```bash
npm test -- --testNamePattern="validation"
```

## Test Coverage

### Current Coverage Areas

#### Search Component
- ✅ Form rendering
- ✅ YouTube URL validation
- ✅ Max comments range validation
- ✅ Model selection
- ✅ API toggle functionality
- ✅ Loading states
- ✅ Error handling (timeout, network, server errors)
- ✅ Successful analysis flow
- ✅ Navigation after analysis

#### Dashboard Component
- ✅ Empty state display
- ✅ User statistics overview
- ✅ Sentiment data rendering
- ✅ Null/undefined safety
- ✅ Confidence statistics
- ✅ Ensemble model information
- ✅ Aspect sentiment display
- ✅ Timeline visualization
- ✅ Top comments display
- ✅ Word clouds
- ✅ Filter statistics
- ✅ Report button visibility
- ✅ API error handling

#### Monitoring Component
- ✅ Analysis list display
- ✅ Summary statistics
- ✅ Empty state
- ✅ Loading states
- ✅ Error handling
- ✅ Refresh functionality
- ✅ Relative time display
- ✅ Dominant sentiment calculation
- ✅ View details navigation
- ✅ Thumbnail fallback handling
- ✅ Model display
- ✅ Last refresh time
- ✅ Error dismissal

#### Report Component
- ✅ Redirect on missing data
- ✅ Complete report rendering
- ✅ Sentiment breakdown
- ✅ Model settings display
- ✅ Ensemble weights
- ✅ Confidence summary
- ✅ Confidence intervals
- ✅ Aspect sentiment table
- ✅ Timeline table
- ✅ Entry limiting (10 aspects, 12 timeline)
- ✅ Missing optional sections
- ✅ Print functionality
- ✅ Back navigation
- ✅ Null value handling

## Writing New Tests

### Basic Test Structure

```javascript
import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import '@testing-library/jest-dom';
import YourComponent from './YourComponent';

describe('YourComponent', () => {
  test('renders correctly', () => {
    render(
      <BrowserRouter>
        <YourComponent />
      </BrowserRouter>
    );

    expect(screen.getByText('Expected Text')).toBeInTheDocument();
  });
});
```

### Testing with AuthContext

```javascript
import AuthContext from '../../context/AuthContext';

const renderWithAuth = (component, authTokenValue = { access: 'test-token' }) => {
  const mockAuthContext = {
    authToken: authTokenValue,
    logoutUser: jest.fn(),
  };

  return render(
    <BrowserRouter>
      <AuthContext.Provider value={mockAuthContext}>
        {component}
      </AuthContext.Provider>
    </BrowserRouter>
  );
};

// Usage
test('example test', () => {
  renderWithAuth(<YourComponent />);
  // ... assertions
});
```

### Testing Async Operations

```javascript
test('fetches data on mount', async () => {
  renderWithAuth(<Dashboard />);

  await waitFor(() => {
    expect(screen.getByText('Expected Data')).toBeInTheDocument();
  });
});
```

### Testing User Interactions

```javascript
test('button click triggers action', async () => {
  renderWithAuth(<Search />);

  const button = screen.getByText('Analyze');
  fireEvent.click(button);

  await waitFor(() => {
    expect(mockNavigate).toHaveBeenCalled();
  });
});
```

### Mocking Axios

```javascript
import axios from 'axios';
jest.mock('axios');

test('handles API response', async () => {
  axios.mockResolvedValue({
    status: 200,
    data: { message: 'Success' },
  });

  // Test code...
});

test('handles API error', async () => {
  axios.mockRejectedValue(new Error('Network error'));

  // Test code...
});
```

## Best Practices

### 1. Test User Behavior, Not Implementation

❌ **Bad**:
```javascript
expect(component.state.isLoading).toBe(true);
```

✅ **Good**:
```javascript
expect(screen.getByText('Loading...')).toBeInTheDocument();
```

### 2. Use Semantic Queries

Prefer queries in this order:
1. `getByRole` - Most accessible
2. `getByLabelText` - For form fields
3. `getByPlaceholderText` - For inputs
4. `getByText` - For non-interactive elements
5. `getByTestId` - Last resort

### 3. Clean Up After Tests

```javascript
beforeEach(() => {
  jest.clearAllMocks();
  localStorage.clear();
});

afterEach(() => {
  localStorage.clear();
});
```

### 4. Test Error States

Always test:
- Loading states
- Error states
- Empty states
- Edge cases (null, undefined, NaN)

### 5. Use Descriptive Test Names

❌ **Bad**:
```javascript
test('it works', () => { ... });
```

✅ **Good**:
```javascript
test('displays error message when YouTube URL is invalid', () => { ... });
```

## Common Testing Patterns

### Testing Forms

```javascript
test('validates form input', async () => {
  renderWithAuth(<Search />);

  const input = screen.getByPlaceholderText('Enter URL');
  const button = screen.getByText('Submit');

  fireEvent.change(input, { target: { value: 'invalid-url' } });
  fireEvent.click(button);

  await waitFor(() => {
    expect(screen.getByText('Invalid URL')).toBeInTheDocument();
  });
});
```

### Testing Navigation

```javascript
const mockNavigate = jest.fn();
jest.mock('react-router-dom', () => ({
  ...jest.requireActual('react-router-dom'),
  useNavigate: () => mockNavigate,
}));

test('navigates on success', async () => {
  // ... trigger navigation

  expect(mockNavigate).toHaveBeenCalledWith('/dashboard', {
    state: expect.any(Object),
  });
});
```

### Testing Conditional Rendering

```javascript
test('shows component when condition is met', () => {
  const data = { hasData: true };
  renderWithContext(<Component />, data);

  expect(screen.getByText('Data Content')).toBeInTheDocument();
});

test('hides component when condition is not met', () => {
  const data = { hasData: false };
  renderWithContext(<Component />, data);

  expect(screen.queryByText('Data Content')).not.toBeInTheDocument();
});
```

## Debugging Tests

### View Rendered Output

```javascript
import { render, screen } from '@testing-library/react';

test('debug example', () => {
  const { debug } = render(<Component />);
  debug(); // Prints DOM to console
});
```

### Use screen.debug()

```javascript
test('debug specific element', () => {
  render(<Component />);
  screen.debug(screen.getByRole('button'));
});
```

### Check What Queries Are Available

```javascript
test('log queries', () => {
  render(<Component />);
  screen.logTestingPlaygroundURL();
});
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Frontend Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v2
      - name: Setup Node
        uses: actions/setup-node@v2
        with:
          node-version: '16'
      - name: Install dependencies
        run: cd frontend && npm install
      - name: Run tests
        run: cd frontend && npm test -- --coverage --watchAll=false
      - name: Upload coverage
        uses: codecov/codecov-action@v2
        with:
          files: ./frontend/coverage/lcov.info
```

## Coverage Goals

Aim for:
- **Statements**: > 80%
- **Branches**: > 75%
- **Functions**: > 80%
- **Lines**: > 80%

### Generate Coverage Report

```bash
npm test -- --coverage --watchAll=false
```

Coverage report will be generated in `frontend/coverage/lcov-report/index.html`

## Troubleshooting

### Common Issues

#### 1. Tests Timing Out

```javascript
// Increase timeout for specific test
test('slow operation', async () => {
  // ... test code
}, 10000); // 10 second timeout
```

#### 2. Act Warnings

Wrap state updates in `waitFor`:

```javascript
await waitFor(() => {
  expect(screen.getByText('Updated')).toBeInTheDocument();
});
```

#### 3. Module Not Found

Check that all required dependencies are installed:

```bash
npm install --save-dev @testing-library/react @testing-library/jest-dom
```

#### 4. Mock Not Working

Ensure mocks are defined before imports:

```javascript
jest.mock('axios');
import axios from 'axios';
```

## Resources

- [React Testing Library Documentation](https://testing-library.com/docs/react-testing-library/intro/)
- [Jest Documentation](https://jestjs.io/docs/getting-started)
- [Testing Library Cheatsheet](https://testing-library.com/docs/react-testing-library/cheatsheet)
- [Common Testing Mistakes](https://kentcdodds.com/blog/common-mistakes-with-react-testing-library)

## Maintenance

### Regular Tasks

1. **Weekly**: Review and update tests for new features
2. **Monthly**: Check coverage reports and improve coverage
3. **Quarterly**: Update testing dependencies
4. **As Needed**: Refactor tests to match code changes

### Code Review Checklist

- [ ] All new components have corresponding tests
- [ ] Tests cover happy path and error cases
- [ ] Tests are descriptive and maintainable
- [ ] No console errors or warnings
- [ ] Coverage meets minimum thresholds
- [ ] Tests are deterministic (no flakiness)
