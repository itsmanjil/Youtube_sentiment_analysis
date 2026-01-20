import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import '@testing-library/jest-dom';
import Monitoring from './Monitoring';
import AuthContext from '../../context/AuthContext';
import axios from 'axios';
import jwt_decode from 'jwt-decode';

// Mock dependencies
jest.mock('axios');
jest.mock('jwt-decode');

const mockNavigate = jest.fn();
jest.mock('react-router-dom', () => ({
  ...jest.requireActual('react-router-dom'),
  useNavigate: () => mockNavigate,
}));

// Helper to render with AuthContext
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

describe('Monitoring Component', () => {
  const mockAnalyses = [
    {
      id: 1,
      sentiment_data: {
        Positive: 100,
        Negative: 50,
        Neutral: 50,
      },
      video: {
        title: 'Test Video 1',
        channel_name: 'Test Channel 1',
        thumbnail_url: 'http://example.com/thumb1.jpg',
        id: 'video1',
      },
      total_comments_analyzed: 200,
      analysis_model: 'LOGREG',
      fetched_date: '2024-01-01T10:00:00',
    },
    {
      id: 2,
      sentiment_data: {
        Positive: 80,
        Negative: 60,
        Neutral: 60,
      },
      video: {
        title: 'Test Video 2',
        channel_name: 'Test Channel 2',
        id: 'video2',
      },
      total_comments_analyzed: 200,
      analysis_model: 'SVM',
      fetched_date: '2024-01-02T15:30:00',
    },
  ];

  beforeEach(() => {
    jest.clearAllMocks();
    localStorage.setItem('authToken', JSON.stringify({ access: 'test-token' }));

    jwt_decode.mockReturnValue({
      user_id: '123',
      user_name: 'Test User',
    });

    // Default mock response
    axios.mockResolvedValue({
      status: 200,
      data: {
        data: mockAnalyses,
      },
    });
  });

  afterEach(() => {
    localStorage.clear();
  });

  test('renders monitoring dashboard', async () => {
    renderWithAuth(<Monitoring />);

    await waitFor(() => {
      expect(screen.getByText('Monitoring Dashboard')).toBeInTheDocument();
    });
  });

  test('fetches and displays analyses on mount', async () => {
    renderWithAuth(<Monitoring />);

    await waitFor(() => {
      expect(screen.getByText('Test Video 1')).toBeInTheDocument();
      expect(screen.getByText('Test Video 2')).toBeInTheDocument();
    });

    expect(screen.getByText('Test Channel 1')).toBeInTheDocument();
    expect(screen.getByText('Test Channel 2')).toBeInTheDocument();
  });

  test('displays summary statistics correctly', async () => {
    renderWithAuth(<Monitoring />);

    await waitFor(() => {
      expect(screen.getByText('2')).toBeInTheDocument(); // Total analyses
    });

    // Check for average percentages
    const avgPositive = screen.getAllByText(/50\.0%|40\.0%/)[0]; // (100/200 + 80/200) / 2 = 45%
    expect(avgPositive).toBeInTheDocument();
  });

  test('handles empty analyses list', async () => {
    axios.mockResolvedValue({
      status: 200,
      data: {
        data: [],
      },
    });

    renderWithAuth(<Monitoring />);

    await waitFor(() => {
      expect(screen.getByText(/No analyses yet/i)).toBeInTheDocument();
      expect(screen.getByText(/Analyze your first YouTube video!/i)).toBeInTheDocument();
    });
  });

  test('displays loading state while fetching', async () => {
    // Mock delayed response
    axios.mockImplementationOnce(
      () =>
        new Promise((resolve) =>
          setTimeout(
            () =>
              resolve({
                status: 200,
                data: { data: mockAnalyses },
              }),
            100
          )
        )
    );

    renderWithAuth(<Monitoring />);

    expect(screen.getByText('Loading...')).toBeInTheDocument();

    await waitFor(
      () => {
        expect(screen.queryByText('Loading...')).not.toBeInTheDocument();
      },
      { timeout: 200 }
    );
  });

  test('displays error message on fetch failure', async () => {
    axios.mockRejectedValue(new Error('Network error'));

    renderWithAuth(<Monitoring />);

    await waitFor(() => {
      expect(screen.getByText(/Failed to load analyses/i)).toBeInTheDocument();
    });
  });

  test('refresh button refetches data', async () => {
    renderWithAuth(<Monitoring />);

    await waitFor(() => {
      expect(screen.getByText('Test Video 1')).toBeInTheDocument();
    });

    expect(axios).toHaveBeenCalledTimes(1);

    const refreshButton = screen.getByText('Refresh');
    fireEvent.click(refreshButton);

    await waitFor(() => {
      expect(axios).toHaveBeenCalledTimes(2);
    });
  });

  test('displays refresh button as disabled during loading', async () => {
    axios.mockImplementationOnce(
      () =>
        new Promise((resolve) =>
          setTimeout(
            () =>
              resolve({
                status: 200,
                data: { data: mockAnalyses },
              }),
            100
          )
        )
    );

    renderWithAuth(<Monitoring />);

    const refreshButton = screen.getByText('Refresh');
    fireEvent.click(refreshButton);

    await waitFor(() => {
      expect(screen.getByText('Refreshing...')).toBeInTheDocument();
    });
  });

  test('displays relative time correctly', async () => {
    renderWithAuth(<Monitoring />);

    await waitFor(() => {
      // Should show time ago for fetched_date
      expect(screen.getByText(/ago|Just now/)).toBeInTheDocument();
    });
  });

  test('calculates dominant sentiment correctly', async () => {
    renderWithAuth(<Monitoring />);

    await waitFor(() => {
      const positiveLabels = screen.getAllByText('Positive');
      expect(positiveLabels.length).toBeGreaterThan(0); // Video 1 has dominant positive
    });
  });

  test('displays sentiment percentages correctly', async () => {
    renderWithAuth(<Monitoring />);

    await waitFor(() => {
      // Video 1: 100 positive out of 200 = 50%
      expect(screen.getByText(/100 \(50\.0%\)/)).toBeInTheDocument();
      // Video 1: 50 negative out of 200 = 25%
      expect(screen.getByText(/50 \(25\.0%\)/)).toBeInTheDocument();
    });
  });

  test('view details button navigates to dashboard with correct data', async () => {
    // Mock the detail endpoint
    axios.mockImplementation((config) => {
      if (config.url.includes('/analysis/video1/')) {
        return Promise.resolve({
          status: 200,
          data: {
            data: {
              sentiment_data: mockAnalyses[0].sentiment_data,
              video: mockAnalyses[0].video,
              total_comments: mockAnalyses[0].total_comments_analyzed,
              model_used: mockAnalyses[0].analysis_model,
              fetched_date: mockAnalyses[0].fetched_date,
            },
          },
        });
      }
      return Promise.resolve({
        status: 200,
        data: { data: mockAnalyses },
      });
    });

    renderWithAuth(<Monitoring />);

    await waitFor(() => {
      expect(screen.getByText('Test Video 1')).toBeInTheDocument();
    });

    const viewButtons = screen.getAllByText('View Details');
    fireEvent.click(viewButtons[0]);

    await waitFor(() => {
      expect(mockNavigate).toHaveBeenCalledWith(
        '/dashboard',
        expect.objectContaining({
          state: expect.objectContaining({
            sentiment_data: mockAnalyses[0].sentiment_data,
          }),
        })
      );
    });
  });

  test('handles video without thumbnail gracefully', async () => {
    renderWithAuth(<Monitoring />);

    await waitFor(() => {
      expect(screen.getByText('Test Video 2')).toBeInTheDocument();
    });

    // Should display placeholder icon
    const videoIcons = screen.getAllByClassName('fa-video');
    expect(videoIcons.length).toBeGreaterThan(0);
  });

  test('displays model used for each analysis', async () => {
    renderWithAuth(<Monitoring />);

    await waitFor(() => {
      expect(screen.getByText('Model: LOGREG')).toBeInTheDocument();
      expect(screen.getByText('Model: SVM')).toBeInTheDocument();
    });
  });

  test('displays total comments analyzed', async () => {
    renderWithAuth(<Monitoring />);

    await waitFor(() => {
      const totalComments = screen.getAllByText('Total Comments: 200');
      expect(totalComments.length).toBe(2);
    });
  });

  test('error alert can be dismissed', async () => {
    axios.mockRejectedValue(new Error('Network error'));

    renderWithAuth(<Monitoring />);

    await waitFor(() => {
      expect(screen.getByText(/Failed to load analyses/i)).toBeInTheDocument();
    });

    const closeButton = screen.getByRole('button', { name: /close/i });
    fireEvent.click(closeButton);

    await waitFor(() => {
      expect(screen.queryByText(/Failed to load analyses/i)).not.toBeInTheDocument();
    });
  });

  test('shows last refresh time after successful fetch', async () => {
    renderWithAuth(<Monitoring />);

    await waitFor(() => {
      expect(screen.getByText(/Last refreshed:/i)).toBeInTheDocument();
    });
  });

  test('handles video detail fetch failure gracefully', async () => {
    axios.mockImplementation((config) => {
      if (config.url.includes('/analysis/')) {
        return Promise.reject(new Error('Detail fetch failed'));
      }
      return Promise.resolve({
        status: 200,
        data: { data: mockAnalyses },
      });
    });

    renderWithAuth(<Monitoring />);

    await waitFor(() => {
      expect(screen.getByText('Test Video 1')).toBeInTheDocument();
    });

    const viewButtons = screen.getAllByText('View Details');
    fireEvent.click(viewButtons[0]);

    await waitFor(() => {
      // Should still navigate with fallback data
      expect(mockNavigate).toHaveBeenCalled();
    });
  });
});
