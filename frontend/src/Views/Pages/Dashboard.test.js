import React from 'react';
import { render, screen, waitFor, within } from '@testing-library/react';
import { BrowserRouter, MemoryRouter } from 'react-router-dom';
import '@testing-library/jest-dom';
import Dashboard from './Dashboard';
import AuthContext from '../../context/AuthContext';
import axios from 'axios';
import jwt_decode from 'jwt-decode';

// Mock dependencies
jest.mock('axios');
jest.mock('jwt-decode');
jest.mock('recharts-to-png', () => ({
  useCurrentPng: () => [jest.fn(), { ref: { current: null } }],
}));

// Helper to render with AuthContext and Router
const renderWithContext = (component, locationState = null, authTokenValue = { access: 'test-token' }) => {
  const mockAuthContext = {
    authToken: authTokenValue,
    logoutUser: jest.fn(),
  };

  return render(
    <MemoryRouter initialEntries={[{ pathname: '/dashboard', state: locationState }]}>
      <AuthContext.Provider value={mockAuthContext}>
        {component}
      </AuthContext.Provider>
    </MemoryRouter>
  );
};

describe('Dashboard Component', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    localStorage.setItem('authToken', JSON.stringify({ access: 'test-token' }));

    // Mock jwt_decode
    jwt_decode.mockReturnValue({
      user_id: '123',
      user_name: 'Test User',
    });

    // Mock default user data response
    axios.mockResolvedValue({
      status: 200,
      data: {
        user_name: 'Test User',
        email: 'test@example.com',
        searched_list: [],
      },
    });
  });

  afterEach(() => {
    localStorage.clear();
  });

  test('renders dashboard without analysis data', async () => {
    renderWithContext(<Dashboard />);

    await waitFor(() => {
      expect(screen.getByText('Dashboard')).toBeInTheDocument();
    });

    expect(screen.getByText(/No analysis yet/i)).toBeInTheDocument();
  });

  test('displays user statistics overview', async () => {
    // Mock analyses response
    axios.mockImplementation((config) => {
      if (config.url.includes('/analyses/')) {
        return Promise.resolve({
          status: 200,
          data: {
            data: [
              {
                sentiment_data: { Positive: 100, Negative: 50, Neutral: 50 },
              },
              {
                sentiment_data: { Positive: 80, Negative: 60, Neutral: 60 },
              },
            ],
          },
        });
      }
      return Promise.resolve({
        status: 200,
        data: {
          user_name: 'Test User',
          email: 'test@example.com',
          searched_list: [],
        },
      });
    });

    renderWithContext(<Dashboard />);

    await waitFor(() => {
      expect(screen.getByText('Your Analysis Overview')).toBeInTheDocument();
    });

    expect(screen.getByText('2')).toBeInTheDocument(); // Total Videos
  });

  test('renders analysis results when data is provided', async () => {
    const analysisData = {
      sentiment_data: {
        Positive: 150,
        Negative: 50,
        Neutral: 100,
      },
      video: {
        title: 'Test Video',
        channel: 'Test Channel',
        view_count: 10000,
        like_count: 500,
      },
      fetched_date: '2024-01-01',
    };

    renderWithContext(<Dashboard />, analysisData);

    await waitFor(() => {
      expect(screen.getByText(/Analysis results for:/i)).toBeInTheDocument();
      expect(screen.getByText('Test Video')).toBeInTheDocument();
    });

    expect(screen.getByText('150')).toBeInTheDocument(); // Positive count
    expect(screen.getByText('50')).toBeInTheDocument(); // Negative count
    expect(screen.getByText('100')).toBeInTheDocument(); // Neutral count
    expect(screen.getByText('300')).toBeInTheDocument(); // Total comments
  });

  test('handles null/undefined sentiment breakdown safely', async () => {
    const analysisData = {
      sentiment_data: null,
      video: {
        title: 'Test Video',
      },
    };

    renderWithContext(<Dashboard />, analysisData);

    await waitFor(() => {
      // Should render without crashing
      expect(screen.getByText('Dashboard')).toBeInTheDocument();
    });
  });

  test('displays confidence statistics when available', async () => {
    const analysisData = {
      sentiment_data: {
        Positive: 100,
        Negative: 50,
        Neutral: 50,
      },
      video: { title: 'Test Video' },
      confidence_stats: {
        mean: 0.85,
        median: 0.90,
        low_confidence_ratio: 0.15,
        threshold: 0.6,
      },
      model_used: 'ensemble',
    };

    renderWithContext(<Dashboard />, analysisData);

    await waitFor(() => {
      expect(screen.getByText('Model & Experiment Settings')).toBeInTheDocument();
      expect(screen.getByText('Confidence & Uncertainty')).toBeInTheDocument();
    });

    expect(screen.getByText('85.0%')).toBeInTheDocument(); // Mean confidence
    expect(screen.getByText('90.0%')).toBeInTheDocument(); // Median confidence
    expect(screen.getByText('15.0%')).toBeInTheDocument(); // Low confidence ratio
  });

  test('displays ensemble model information', async () => {
    const analysisData = {
      sentiment_data: {
        Positive: 100,
        Negative: 50,
        Neutral: 50,
      },
      video: { title: 'Test Video' },
      model_used: 'CI Ensemble',
      analysis_meta: {
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

    renderWithContext(<Dashboard />, analysisData);

    await waitFor(() => {
      expect(screen.getByText('Model & Experiment Settings')).toBeInTheDocument();
    });

    expect(screen.getByText(/LogReg, SVM, Hybrid-DL/i)).toBeInTheDocument();
    expect(screen.getByText(/1000/i)).toBeInTheDocument(); // Bootstrap samples
  });

  test('displays aspect sentiment table when available', async () => {
    const analysisData = {
      sentiment_data: {
        Positive: 100,
        Negative: 50,
        Neutral: 50,
      },
      video: { title: 'Test Video' },
      aspect_sentiment: [
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
    };

    renderWithContext(<Dashboard />, analysisData);

    await waitFor(() => {
      expect(screen.getByText('Aspect Sentiment')).toBeInTheDocument();
    });

    expect(screen.getByText('video quality')).toBeInTheDocument();
    expect(screen.getByText('audio')).toBeInTheDocument();
    expect(screen.getByText('50')).toBeInTheDocument(); // Aspect count
  });

  test('displays sentiment timeline when available', async () => {
    const analysisData = {
      sentiment_data: {
        Positive: 100,
        Negative: 50,
        Neutral: 50,
      },
      video: { title: 'Test Video' },
      sentiment_timeline: {
        '2024-01-01T10:00:00': { Positive: 10, Negative: 5, Neutral: 5 },
        '2024-01-01T11:00:00': { Positive: 15, Negative: 8, Neutral: 7 },
      },
    };

    renderWithContext(<Dashboard />, analysisData);

    await waitFor(() => {
      expect(screen.getByText(/Sentiment Timeline/i)).toBeInTheDocument();
    });
  });

  test('displays top comments when available', async () => {
    const analysisData = {
      sentiment_data: {
        Positive: 100,
        Negative: 50,
        Neutral: 50,
      },
      video: { title: 'Test Video' },
      like_weighted_sentiment: [
        {
          text: 'Great video!',
          author: 'User1',
          likes: 100,
          sentiment: 'Positive',
        },
        {
          text: 'Not good',
          author: 'User2',
          likes: 50,
          sentiment: 'Negative',
        },
      ],
    };

    renderWithContext(<Dashboard />, analysisData);

    await waitFor(() => {
      expect(screen.getByText('Most Influential Comments (By Likes)')).toBeInTheDocument();
    });

    expect(screen.getByText(/Great video!/i)).toBeInTheDocument();
    expect(screen.getByText(/Not good/i)).toBeInTheDocument();
  });

  test('displays word clouds when available', async () => {
    const analysisData = {
      sentiment_data: {
        Positive: 100,
        Negative: 50,
        Neutral: 50,
      },
      video: { title: 'Test Video' },
      top_words_positive: [
        { word: 'great', count: 50 },
        { word: 'amazing', count: 30 },
      ],
      top_words_negative: [
        { word: 'bad', count: 40 },
        { word: 'worst', count: 25 },
      ],
    };

    renderWithContext(<Dashboard />, analysisData);

    await waitFor(() => {
      expect(screen.getByText('Most Frequent Words in Comments')).toBeInTheDocument();
    });

    expect(screen.getByText('great')).toBeInTheDocument();
    expect(screen.getByText('amazing')).toBeInTheDocument();
    expect(screen.getByText('bad')).toBeInTheDocument();
    expect(screen.getByText('worst')).toBeInTheDocument();
  });

  test('displays filter statistics when available', async () => {
    const analysisData = {
      sentiment_data: {
        Positive: 100,
        Negative: 50,
        Neutral: 50,
      },
      video: { title: 'Test Video' },
      filtered: {
        total: 50,
        spam: 20,
        language: 15,
        short: 15,
      },
    };

    renderWithContext(<Dashboard />, analysisData);

    await waitFor(() => {
      expect(screen.getByText('Data Quality & Filtering Statistics')).toBeInTheDocument();
    });

    expect(screen.getByText('20')).toBeInTheDocument(); // Spam removed
    expect(screen.getByText('15')).toBeInTheDocument(); // Non-English
  });

  test('handles API errors gracefully', async () => {
    axios.mockRejectedValue(new Error('API Error'));

    renderWithContext(<Dashboard />);

    await waitFor(() => {
      // Should not crash, just show empty state
      expect(screen.getByText('Dashboard')).toBeInTheDocument();
    });
  });

  test('formatPercent function handles edge cases', async () => {
    const analysisData = {
      sentiment_data: {
        Positive: 100,
        Negative: 50,
        Neutral: 50,
      },
      video: { title: 'Test Video' },
      confidence_stats: {
        mean: null,
        median: undefined,
        low_confidence_ratio: NaN,
      },
    };

    renderWithContext(<Dashboard />, analysisData);

    await waitFor(() => {
      // Should display 0% for null/undefined/NaN values
      const confidenceSection = screen.getByText('Confidence & Uncertainty').closest('.card-body');
      const percentTexts = within(confidenceSection).getAllByText('0%');
      expect(percentTexts.length).toBeGreaterThan(0);
    });
  });

  test('shows report button when analysis data is present', async () => {
    const analysisData = {
      sentiment_data: {
        Positive: 100,
        Negative: 50,
        Neutral: 50,
      },
      video: { title: 'Test Video' },
    };

    renderWithContext(<Dashboard />, analysisData);

    await waitFor(() => {
      const reportButton = screen.getByText('Report');
      expect(reportButton).toBeInTheDocument();
    });
  });

  test('hides report button when no analysis data', async () => {
    renderWithContext(<Dashboard />);

    await waitFor(() => {
      const reportButton = screen.queryByText('Report');
      expect(reportButton).not.toBeInTheDocument();
    });
  });
});
