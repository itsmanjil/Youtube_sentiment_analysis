import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import Search from './Search';
import AuthContext from '../../context/AuthContext';
import axios from 'axios';
import { vi } from 'vitest';

// Mock axios
vi.mock('axios');

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
      axios.mockResolvedValueOnce({
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
      expect(screen.getByText(/Max comments must be between 1 and 1000/i)).toBeInTheDocument();
    });

    // Test too high
    fireEvent.change(maxCommentsInput, { target: { value: '1001' } });
    fireEvent.click(screen.getByDisplayValue('Analyze Video'));

    await waitFor(() => {
      expect(screen.getByText(/Max comments must be between 1 and 1000/i)).toBeInTheDocument();
    });
  });

  test('shows loading state during analysis', async () => {
    renderWithAuth(<Search />);

    const urlInput = screen.getByPlaceholderText('https://www.youtube.com/watch?v=...');
    fireEvent.change(urlInput, { target: { value: 'https://www.youtube.com/watch?v=test' } });

    // Mock delayed response
    axios.mockImplementationOnce(() =>
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
    axios.mockRejectedValueOnce({
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
    axios.mockRejectedValueOnce({
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
    axios.mockRejectedValueOnce({
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

    expect(apiCheckbox).not.toBeChecked();

    fireEvent.click(apiCheckbox);
    expect(apiCheckbox).toBeChecked();

    fireEvent.click(apiCheckbox);
    expect(apiCheckbox).not.toBeChecked();
  });

  test('submits with correct data format', async () => {
    renderWithAuth(<Search />);

    const urlInput = screen.getByPlaceholderText('https://www.youtube.com/watch?v=...');
    const maxCommentsInput = screen.getByLabelText(/Max Comments/i);
    const modelSelect = screen.getByLabelText(/Sentiment Model/i);
    const apiCheckbox = screen.getByLabelText(/Use YouTube API/i);

    fireEvent.change(urlInput, { target: { value: 'https://www.youtube.com/watch?v=test123' } });
    fireEvent.change(maxCommentsInput, { target: { value: '500' } });
    fireEvent.change(modelSelect, { target: { value: 'ensemble' } });
    fireEvent.click(apiCheckbox);

    axios.mockResolvedValueOnce({
      data: {
        sentiment_data: { Positive: 10, Negative: 5, Neutral: 3 },
        video: { title: 'Test Video' },
      },
    });

    fireEvent.click(screen.getByDisplayValue('Analyze Video'));

    await waitFor(() => {
      expect(axios).toHaveBeenCalledWith(
        expect.objectContaining({
          method: 'POST',
          url: 'http://127.0.0.1:8000/api/youtube/analyze/',
          data: {
            video_url: 'https://www.youtube.com/watch?v=test123',
            max_comments: 500,
            use_api: true,
            sentiment_model: 'ensemble',
          },
        })
      );
    });
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

    axios.mockResolvedValueOnce(mockResponse);

    fireEvent.click(screen.getByDisplayValue('Analyze Video'));

    await waitFor(() => {
      expect(mockNavigate).toHaveBeenCalledWith('/dashboard', {
        state: mockResponse.data,
      });
    });
  });
});
