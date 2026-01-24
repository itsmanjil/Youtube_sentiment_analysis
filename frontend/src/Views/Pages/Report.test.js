import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import Report from './Report';
import AuthContext from '../../context/AuthContext';
import { vi } from 'vitest';

// Mock navigate
const mockNavigate = vi.fn();
vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual('react-router-dom');
  return {
    ...actual,
    useNavigate: () => mockNavigate,
  };
});

// Mock CSS import
vi.mock('./report.css', () => ({}));

// Helper to render with AuthContext and Router
const renderWithContext = (locationState = null, authTokenValue = { access: 'test-token' }) => {
  const mockAuthContext = {
    authToken: authTokenValue,
    logoutUser: vi.fn(),
  };

  return render(
    <MemoryRouter initialEntries={[{ pathname: '/report/test', state: locationState }]}>
      <AuthContext.Provider value={mockAuthContext}>
        <Report />
      </AuthContext.Provider>
    </MemoryRouter>
  );
};

describe('Report Component', () => {
  const mockReportData = {
    user_name: 'Test User',
    videoTitle: 'Test Video Analysis',
    fetchedDate: '2024-01-15',
    sentimentBreakdown: [
      { name: 'Negative', value: 50 },
      { name: 'Neutral', value: 100 },
      { name: 'Positive', value: 150 },
    ],
    confidenceStats: {
      mean: 0.85,
      median: 0.90,
      low_confidence_ratio: 0.15,
      threshold: 0.6,
    },
    confidenceIntervals: {
      Positive: { lower: 0.45, upper: 0.55 },
      Neutral: { lower: 0.30, upper: 0.40 },
      Negative: { lower: 0.15, upper: 0.20 },
    },
    aspectSentiment: [
      {
        aspect: 'video quality',
        count: 50,
        ratio: { Positive: 0.8, Neutral: 0.1, Negative: 0.1 },
      },
      {
        aspect: 'audio',
        count: 30,
        ratio: { Positive: 0.6, Neutral: 0.2, Negative: 0.2 },
      },
    ],
    sentimentTimeline: [
      { time: '10:00', positive: 10, neutral: 5, negative: 5 },
      { time: '11:00', positive: 15, neutral: 8, negative: 7 },
      { time: '12:00', positive: 12, neutral: 6, negative: 4 },
    ],
    modelUsed: 'CI Ensemble',
    analysisMeta: {
      ensemble: {
        models: ['LogReg', 'SVM', 'Hybrid-DL'],
        weights: {
          logreg: 0.3,
          svm: 0.4,
          hybrid: 0.3,
        },
      },
      bootstrap_samples: 1000,
      random_seed: 42,
    },
  };

  beforeEach(() => {
    vi.clearAllMocks();
    global.print = vi.fn();
  });

  test('redirects to dashboard when no state is provided', () => {
    renderWithContext(null);

    expect(screen.getByText(/No report data available/i)).toBeInTheDocument();
  });

  test('renders report with all data sections', () => {
    renderWithContext(mockReportData);

    expect(screen.getByText('YouTube Sentiment Report')).toBeInTheDocument();
    expect(screen.getByText('Test Video Analysis')).toBeInTheDocument();
    expect(screen.getByText('Test User')).toBeInTheDocument();
    expect(screen.getByText('300')).toBeInTheDocument(); // Total comments
  });

  test('displays sentiment breakdown table', () => {
    renderWithContext(mockReportData);

    expect(screen.getByText('Negative')).toBeInTheDocument();
    expect(screen.getByText('Neutral')).toBeInTheDocument();
    expect(screen.getByText('Positive')).toBeInTheDocument();
    expect(screen.getByText('50')).toBeInTheDocument();
    expect(screen.getByText('100')).toBeInTheDocument();
    expect(screen.getByText('150')).toBeInTheDocument();
  });

  test('displays model and experiment settings', () => {
    renderWithContext(mockReportData);

    expect(screen.getByText('Model & Experiment Settings')).toBeInTheDocument();
    expect(screen.getByText('CI Ensemble')).toBeInTheDocument();
    expect(screen.getByText('LogReg, SVM, Hybrid-DL')).toBeInTheDocument();
    expect(screen.getByText('1000')).toBeInTheDocument(); // Bootstrap samples
    expect(screen.getByText('42')).toBeInTheDocument(); // Random seed
  });

  test('displays ensemble weights correctly', () => {
    renderWithContext(mockReportData);

    expect(screen.getByText(/logreg: 0\.3/i)).toBeInTheDocument();
    expect(screen.getByText(/svm: 0\.4/i)).toBeInTheDocument();
    expect(screen.getByText(/hybrid: 0\.3/i)).toBeInTheDocument();
  });

  test('displays confidence summary', () => {
    renderWithContext(mockReportData);

    expect(screen.getByText('Confidence Summary')).toBeInTheDocument();
    expect(screen.getByText('85.0%')).toBeInTheDocument(); // Mean
    expect(screen.getByText('90.0%')).toBeInTheDocument(); // Median
    expect(screen.getByText('15.0%')).toBeInTheDocument(); // Low confidence ratio
  });

  test('displays confidence intervals table', () => {
    renderWithContext(mockReportData);

    expect(screen.getByText('Sentiment Confidence Intervals (95%)')).toBeInTheDocument();
    expect(screen.getByText('45.0%')).toBeInTheDocument(); // Positive lower
    expect(screen.getByText('55.0%')).toBeInTheDocument(); // Positive upper
  });

  test('displays aspect sentiment table', () => {
    renderWithContext(mockReportData);

    expect(screen.getByText(/Aspect Sentiment/i)).toBeInTheDocument();
    expect(screen.getByText('video quality')).toBeInTheDocument();
    expect(screen.getByText('audio')).toBeInTheDocument();
    expect(screen.getByText('80.0%')).toBeInTheDocument(); // Positive ratio for video quality
  });

  test('displays sentiment timeline table', () => {
    renderWithContext(mockReportData);

    expect(screen.getByText(/Sentiment Timeline/i)).toBeInTheDocument();
    expect(screen.getByText('10:00')).toBeInTheDocument();
    expect(screen.getByText('11:00')).toBeInTheDocument();
    expect(screen.getByText('12:00')).toBeInTheDocument();
  });

  test('limits timeline display to first 12 entries', () => {
    const dataWithManyTimeline = {
      ...mockReportData,
      sentimentTimeline: Array.from({ length: 20 }, (_, i) => ({
        time: `${i}:00`,
        positive: 10,
        neutral: 5,
        negative: 5,
      })),
    };

    renderWithContext(dataWithManyTimeline);

    // Should only display first 12
    expect(screen.getByText('0:00')).toBeInTheDocument();
    expect(screen.getByText('11:00')).toBeInTheDocument();
    expect(screen.queryByText('12:00')).not.toBeInTheDocument();
  });

  test('limits aspect sentiment display to first 10 entries', () => {
    const dataWithManyAspects = {
      ...mockReportData,
      aspectSentiment: Array.from({ length: 15 }, (_, i) => ({
        aspect: `aspect ${i}`,
        count: 10,
        ratio: { Positive: 0.5, Neutral: 0.3, Negative: 0.2 },
      })),
    };

    renderWithContext(dataWithManyAspects);

    expect(screen.getByText('aspect 0')).toBeInTheDocument();
    expect(screen.getByText('aspect 9')).toBeInTheDocument();
    expect(screen.queryByText('aspect 10')).not.toBeInTheDocument();
  });

  test('handles missing optional sections gracefully', () => {
    const minimalData = {
      user_name: 'Test User',
      videoTitle: 'Test Video',
      fetchedDate: '2024-01-15',
      sentimentBreakdown: [
        { name: 'Negative', value: 50 },
        { name: 'Neutral', value: 100 },
        { name: 'Positive', value: 150 },
      ],
    };

    renderWithContext(minimalData);

    expect(screen.getByText('YouTube Sentiment Report')).toBeInTheDocument();
    expect(screen.queryByText('Model & Experiment Settings')).not.toBeInTheDocument();
    expect(screen.queryByText('Confidence Summary')).not.toBeInTheDocument();
    expect(screen.queryByText('Aspect Sentiment')).not.toBeInTheDocument();
  });

  test('print button triggers window.print', () => {
    window.print = vi.fn();

    renderWithContext(mockReportData);

    const printButton = screen.getByText('Print');
    fireEvent.click(printButton);

    expect(window.print).toHaveBeenCalled();
  });

  test('back button links to dashboard', () => {
    renderWithContext(mockReportData);

    const backButton = screen.getByRole('link', { name: '' }); // The arrow icon
    expect(backButton).toHaveAttribute('href', '/dashboard');
  });

  test('displays correct note with comment counts', () => {
    renderWithContext(mockReportData);

    expect(
      screen.getByText(/processed 300 comments/i)
    ).toBeInTheDocument();
    expect(
      screen.getByText(/150 positive/i)
    ).toBeInTheDocument();
    expect(
      screen.getByText(/50 negative/i)
    ).toBeInTheDocument();
    expect(
      screen.getByText(/100 neutral/i)
    ).toBeInTheDocument();
  });

  test('formatPercent handles null values', () => {
    const dataWithNullValues = {
      ...mockReportData,
      confidenceStats: {
        mean: null,
        median: undefined,
        low_confidence_ratio: NaN,
      },
    };

    renderWithContext(dataWithNullValues);

    const zeroPercentTexts = screen.getAllByText('0%');
    expect(zeroPercentTexts.length).toBeGreaterThan(0);
  });

  test('displays report number', () => {
    renderWithContext(mockReportData);

    expect(screen.getByText(/Report no\./i)).toBeInTheDocument();
    expect(screen.getByText('001')).toBeInTheDocument();
  });

  test('displays searched by information', () => {
    renderWithContext(mockReportData);

    expect(screen.getByText('Searched By:')).toBeInTheDocument();
    expect(screen.getByText('Test User')).toBeInTheDocument();
  });

  test('displays video analyzed information', () => {
    renderWithContext(mockReportData);

    expect(screen.getByText('Video analyzed:')).toBeInTheDocument();
    expect(screen.getByText('Test Video Analysis')).toBeInTheDocument();
  });

  test('displays analysis date', () => {
    renderWithContext(mockReportData);

    expect(screen.getByText('Date:')).toBeInTheDocument();
    expect(screen.getByText('2024-01-15')).toBeInTheDocument();
  });

  test('handles missing user name gracefully', () => {
    const dataWithoutUsername = {
      ...mockReportData,
      user_name: undefined,
    };

    renderWithContext(dataWithoutUsername);

    expect(screen.getByText('Unknown User')).toBeInTheDocument();
  });

  test('print and back buttons have correct styling classes', () => {
    renderWithContext(mockReportData);

    const printButton = screen.getByText('Print');
    expect(printButton).toHaveClass('btn', 'btn-dark');
  });

  test('renders all sections with print-friendly classes', () => {
    renderWithContext(mockReportData);

    // Check that d-print-none class exists on navigation elements
    const navWrapper = document.querySelector('.container-nav-wrapper');
    expect(navWrapper).toHaveClass('d-print-none');
  });
});
