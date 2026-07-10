import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import Search from './Search';
import AuthContext from '../../context/AuthContext';
import axiosInstance from '../../axios';
import { vi } from 'vitest';

// Mock axios instance
vi.mock('../../axios', () => ({
  default: vi.fn(),
}));

// Mock navigate
const mockNavigate = vi.fn();
vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual('react-router-dom');
  return {
    ...actual,
    useNavigate: () => mockNavigate,
  };
});

// Helper to render with AuthContext
const renderWithAuth = (component, authTokenValue = { access: 'test-token' }) => {
  const mockAuthContext = {
    authToken: authTokenValue,
    logoutUser: vi.fn(),
  };

  return render(
    <BrowserRouter>
      <AuthContext.Provider value={mockAuthContext}>
        {component}
      </AuthContext.Provider>
    </BrowserRouter>
  );
};

describe('Search Component', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.setItem('authToken', JSON.stringify({ access: 'test-token' }));
  });

  afterEach(() => {
    localStorage.clear();
  });

  test('renders search form correctly', () => {
    renderWithAuth(<Search />);

    expect(screen.getByText('Analyze YouTube Video')).toBeInTheDocument();
    expect(screen.getByPlaceholderText('https://www.youtube.com/watch?v=...')).toBeInTheDocument();
    expect(screen.getByLabelText(/Max Comments/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/Sentiment Model/i)).toBeInTheDocument();
  });

  test('shows error when video URL is empty', async () => {
    renderWithAuth(<Search />);

    const analyzeButton = screen.getByDisplayValue('Analyze Video');
    fireEvent.click(analyzeButton);

    await waitFor(() => {
      expect(screen.getByText('YouTube URL is required')).toBeInTheDocument();
    });
  });

  test('shows error for invalid YouTube URL format', async () => {
    renderWithAuth(<Search />);

    const urlInput = screen.getByPlaceholderText('https://www.youtube.com/watch?v=...');
    fireEvent.change(urlInput, { target: { value: 'https://invalid-url.com' } });

    const analyzeButton = screen.getByDisplayValue('Analyze Video');
    fireEvent.click(analyzeButton);

    await waitFor(() => {
      expect(screen.getByText(/Invalid YouTube URL format/i)).toBeInTheDocument();
    });
  });

  test('accepts valid YouTube URL formats', async () => {
    renderWithAuth(<Search />);

    const validUrls = [
      'https://www.youtube.com/watch?v=dQw4w9WgXcQ',
      'https://youtu.be/dQw4w9WgXcQ',
      'http://youtube.com/watch?v=dQw4w9WgXcQ',
      'www.youtube.com/watch?v=dQw4w9WgXcQ',
    ];

    for (const url of validUrls) {
      const urlInput = screen.getByPlaceholderText('https://www.youtube.com/watch?v=...');
      fireEvent.change(urlInput, { target: { value: url } });

      const analyzeButton = screen.getByDisplayValue('Analyze Video');

      // Mock successful response
      axiosInstance.mockResolvedValueOnce({
        data: {
          sentiment_data: { Positive: 10, Negative: 5, Neutral: 3 },
          video: { title: 'Test Video' },
        },
      });

      fireEvent.click(analyzeButton);

      await waitFor(() => {
        expect(mockNavigate).toHaveBeenCalled();
      });

      vi.clearAllMocks();
    }
  });

  test('validates max comments range', async () => {
    renderWithAuth(<Search />);

    const urlInput = screen.getByPlaceholderText('https://www.youtube.com/watch?v=...');
    fireEvent.change(urlInput, { target: { value: 'https://www.youtube.com/watch?v=test' } });

    const maxCommentsInput = screen.getByLabelText(/Max Comments/i);

    // Test too low
    fireEvent.change(maxCommentsInput, { target: { value: '0' } });
    fireEvent.click(screen.getByDisplayValue('Analyze Video'));

    await waitFor(() => {
      expect(screen.getByText(/Max comments must be between 1 and 2000/i)).toBeInTheDocument();
    });

    // Test too high
    fireEvent.change(maxCommentsInput, { target: { value: '2001' } });
    fireEvent.click(screen.getByDisplayValue('Analyze Video'));

    await waitFor(() => {
      expect(screen.getByText(/Max comments must be between 1 and 2000/i)).toBeInTheDocument();
    });
  });

  test('shows loading state during analysis', async () => {
    renderWithAuth(<Search />);

    const urlInput = screen.getByPlaceholderText('https://www.youtube.com/watch?v=...');
    fireEvent.change(urlInput, { target: { value: 'https://www.youtube.com/watch?v=test' } });

    // Mock delayed response
    axiosInstance.mockImplementationOnce(() =>
      new Promise(resolve => setTimeout(() => resolve({
        data: {
          sentiment_data: { Positive: 10, Negative: 5, Neutral: 3 },
          video: { title: 'Test Video' },
        },
      }), 100))
    );

    const analyzeButton = screen.getByDisplayValue('Analyze Video');
    fireEvent.click(analyzeButton);

    // Should show loading state
    await waitFor(() => {
      expect(screen.getByDisplayValue('Analyzing...')).toBeInTheDocument();
    });
  });

  test('handles server error responses', async () => {
    renderWithAuth(<Search />);

    const urlInput = screen.getByPlaceholderText('https://www.youtube.com/watch?v=...');
    fireEvent.change(urlInput, { target: { value: 'https://www.youtube.com/watch?v=test' } });

    // Mock server error
    axiosInstance.mockRejectedValueOnce({
      response: { status: 500, data: { message: 'Server error' } },
    });

    fireEvent.click(screen.getByDisplayValue('Analyze Video'));

    await waitFor(() => {
      expect(screen.getByText(/Server error/i)).toBeInTheDocument();
    });
  });

  test('handles network error', async () => {
    renderWithAuth(<Search />);

    const urlInput = screen.getByPlaceholderText('https://www.youtube.com/watch?v=...');
    fireEvent.change(urlInput, { target: { value: 'https://www.youtube.com/watch?v=test' } });

    // Mock network error
    axiosInstance.mockRejectedValueOnce({
      request: {},
    });

    fireEvent.click(screen.getByDisplayValue('Analyze Video'));

    await waitFor(() => {
      expect(screen.getByText(/Cannot connect to server/i)).toBeInTheDocument();
    });
  });

  test('handles timeout error', async () => {
    renderWithAuth(<Search />);

    const urlInput = screen.getByPlaceholderText('https://www.youtube.com/watch?v=...');
    fireEvent.change(urlInput, { target: { value: 'https://www.youtube.com/watch?v=test' } });

    // Mock timeout error
    axiosInstance.mockRejectedValueOnce({
      code: 'ECONNABORTED',
    });

    fireEvent.click(screen.getByDisplayValue('Analyze Video'));

    await waitFor(() => {
      expect(screen.getByText(/Request timeout/i)).toBeInTheDocument();
    });
  });

  test('changes sentiment model selection', () => {
    renderWithAuth(<Search />);

    const modelSelect = screen.getByLabelText(/Sentiment Model/i);

    fireEvent.change(modelSelect, { target: { value: 'svm' } });
    expect(modelSelect.value).toBe('svm');

    fireEvent.change(modelSelect, { target: { value: 'ensemble' } });
    expect(modelSelect.value).toBe('ensemble');
  });

  test('toggles API usage checkbox', () => {
    renderWithAuth(<Search />);

    const apiCheckbox = screen.getByLabelText(/Use YouTube API/i);

    expect(apiCheckbox).toBeChecked();

    fireEvent.click(apiCheckbox);
    expect(apiCheckbox).not.toBeChecked();

    fireEvent.click(apiCheckbox);
    expect(apiCheckbox).toBeChecked();
  });

  test('submits with correct data format', async () => {
    renderWithAuth(<Search />);

    const urlInput = screen.getByPlaceholderText('https://www.youtube.com/watch?v=...');
    const maxCommentsInput = screen.getByLabelText(/Max Comments/i);
    const modelSelect = screen.getByLabelText(/Sentiment Model/i);

    fireEvent.change(urlInput, { target: { value: 'https://www.youtube.com/watch?v=test123' } });
    fireEvent.change(maxCommentsInput, { target: { value: '500' } });
    fireEvent.change(modelSelect, { target: { value: 'ensemble' } });

    axiosInstance.mockResolvedValueOnce({
      data: {
        sentiment_data: { Positive: 10, Negative: 5, Neutral: 3 },
        video: { title: 'Test Video' },
      },
    });

    fireEvent.click(screen.getByDisplayValue('Analyze Video'));

    await waitFor(() => {
      expect(axiosInstance).toHaveBeenCalledWith(
        expect.objectContaining({
          method: 'POST',
          url: 'youtube/analyze/',
          data: expect.objectContaining({
            video_url: 'https://www.youtube.com/watch?v=test123',
            max_comments: 500,
            use_api: true,
            sentiment_model: 'ensemble',
          }),
        })
      );
    });

    const payload = axiosInstance.mock.calls[0][0].data;
    expect(payload).not.toHaveProperty('meta_learner_path');
  });

  test('navigates to dashboard on successful analysis', async () => {
    renderWithAuth(<Search />);

    const urlInput = screen.getByPlaceholderText('https://www.youtube.com/watch?v=...');
    fireEvent.change(urlInput, { target: { value: 'https://www.youtube.com/watch?v=test' } });

    const mockResponse = {
      data: {
        sentiment_data: { Positive: 10, Negative: 5, Neutral: 3 },
        video: { title: 'Test Video' },
      },
    };

    axiosInstance.mockResolvedValueOnce(mockResponse);

    fireEvent.click(screen.getByDisplayValue('Analyze Video'));

    await waitFor(() => {
      expect(mockNavigate).toHaveBeenCalledWith('/dashboard', {
        state: mockResponse.data,
      });
    });
  });

  test('polls the background job and navigates once it completes (202 + job_id path)', async () => {
    // youtube/analyze/ runs in the background in real deployments and
    // returns 202 + a job_id immediately; the frontend must poll
    // youtube/analyze/status/<id>/ until it's done rather than treating the
    // initial response as the result.
    vi.useFakeTimers({ shouldAdvanceTime: true });
    try {
      renderWithAuth(<Search />);

      const urlInput = screen.getByPlaceholderText('https://www.youtube.com/watch?v=...');
      fireEvent.change(urlInput, { target: { value: 'https://www.youtube.com/watch?v=test' } });

      const finalResult = {
        status: 'done',
        sentiment_data: { Positive: 4, Negative: 1, Neutral: 2 },
        video: { title: 'Polled Video' },
      };

      axiosInstance
        .mockResolvedValueOnce({ status: 202, data: { job_id: 42, status: 'pending' } })
        .mockResolvedValueOnce({ status: 200, data: { status: 'running' } })
        .mockResolvedValueOnce({ status: 200, data: finalResult });

      await act(async () => {
        fireEvent.click(screen.getByDisplayValue('Analyze Video'));
      });

      // Let the two poll iterations' setTimeout delays elapse.
      await act(async () => {
        await vi.advanceTimersByTimeAsync(2000);
      });
      await act(async () => {
        await vi.advanceTimersByTimeAsync(2000);
      });

      await vi.waitFor(() => {
        expect(mockNavigate).toHaveBeenCalledWith('/dashboard', {
          state: finalResult,
        });
      });

      expect(axiosInstance).toHaveBeenCalledWith(
        expect.objectContaining({ method: 'GET', url: 'youtube/analyze/status/42/' })
      );
    } finally {
      vi.useRealTimers();
    }
  });

  test('shows the failure message when the background job fails', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    try {
      renderWithAuth(<Search />);

      const urlInput = screen.getByPlaceholderText('https://www.youtube.com/watch?v=...');
      fireEvent.change(urlInput, { target: { value: 'https://www.youtube.com/watch?v=test' } });

      axiosInstance
        .mockResolvedValueOnce({ status: 202, data: { job_id: 7, status: 'pending' } })
        .mockResolvedValueOnce({
          status: 404,
          data: { status: 'failed', msg: 'Video not found. It may be private, deleted, or the URL is incorrect.' },
        });

      await act(async () => {
        fireEvent.click(screen.getByDisplayValue('Analyze Video'));
      });

      await act(async () => {
        await vi.advanceTimersByTimeAsync(2000);
      });

      await vi.waitFor(() => {
        expect(
          screen.getByText(/Video not found\. It may be private, deleted, or the URL is incorrect\./i)
        ).toBeInTheDocument();
      });
      expect(mockNavigate).not.toHaveBeenCalled();
    } finally {
      vi.useRealTimers();
    }
  });

  describe('video search picker', () => {
    test('search button is disabled until a query is entered', () => {
      renderWithAuth(<Search />);

      const searchButton = screen.getByRole('button', { name: 'Search' });
      expect(searchButton).toBeDisabled();

      const searchInput = screen.getByPlaceholderText('Search YouTube by title or keyword...');
      fireEvent.change(searchInput, { target: { value: 'rick astley' } });
      expect(searchButton).not.toBeDisabled();
    });

    test('searches YouTube and displays results', async () => {
      renderWithAuth(<Search />);

      const searchInput = screen.getByPlaceholderText('Search YouTube by title or keyword...');
      fireEvent.change(searchInput, { target: { value: 'rick astley' } });

      axiosInstance.mockResolvedValueOnce({
        status: 200,
        data: {
          data: [
            {
              video_id: 'dQw4w9WgXcQ',
              title: 'Rick Astley - Never Gonna Give You Up',
              channel: 'Rick Astley',
              thumbnail_url: 'https://i.ytimg.com/vi/dQw4w9WgXcQ/mqdefault.jpg',
            },
          ],
        },
      });

      fireEvent.click(screen.getByRole('button', { name: 'Search' }));

      await waitFor(() => {
        expect(axiosInstance).toHaveBeenCalledWith(
          expect.objectContaining({
            method: 'GET',
            url: 'youtube/search/',
            params: { q: 'rick astley', max_results: 8 },
          })
        );
      });

      expect(await screen.findByText('Rick Astley - Never Gonna Give You Up')).toBeInTheDocument();
    });

    test('picking a result fills in the video URL and shows a confirmation', async () => {
      renderWithAuth(<Search />);

      const searchInput = screen.getByPlaceholderText('Search YouTube by title or keyword...');
      fireEvent.change(searchInput, { target: { value: 'rick astley' } });

      axiosInstance.mockResolvedValueOnce({
        status: 200,
        data: {
          data: [
            {
              video_id: 'dQw4w9WgXcQ',
              title: 'Rick Astley - Never Gonna Give You Up',
              channel: 'Rick Astley',
              thumbnail_url: null,
            },
          ],
        },
      });

      fireEvent.click(screen.getByRole('button', { name: 'Search' }));

      const resultButton = await screen.findByText('Rick Astley - Never Gonna Give You Up');
      fireEvent.click(resultButton);

      const urlInput = screen.getByPlaceholderText('https://www.youtube.com/watch?v=...');
      expect(urlInput.value).toBe('https://www.youtube.com/watch?v=dQw4w9WgXcQ');
      expect(screen.getByText(/Selected:/i)).toBeInTheDocument();
      // The result list collapses once a video is picked.
      expect(screen.queryByText('Rick Astley', { selector: '.text-muted' })).not.toBeInTheDocument();
    });

    test('shows an error message when the search request fails', async () => {
      renderWithAuth(<Search />);

      const searchInput = screen.getByPlaceholderText('Search YouTube by title or keyword...');
      fireEvent.change(searchInput, { target: { value: 'rick astley' } });

      axiosInstance.mockResolvedValueOnce({
        status: 429,
        data: { msg: 'YouTube API daily quota exceeded. Please try again tomorrow.' },
      });

      fireEvent.click(screen.getByRole('button', { name: 'Search' }));

      expect(
        await screen.findByText('YouTube API daily quota exceeded. Please try again tomorrow.')
      ).toBeInTheDocument();
    });
  });
});
