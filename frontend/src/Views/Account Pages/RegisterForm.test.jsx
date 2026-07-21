import { render, screen, fireEvent } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { vi } from 'vitest';
import RegisterForm from './RegisterForm';
import axiosInstance from '../../axios';

vi.mock('../../axios', () => ({
  default: { post: vi.fn() },
}));

vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual('react-router-dom');
  return {
    ...actual,
    useNavigate: () => vi.fn(),
  };
});

const renderForm = () => render(<BrowserRouter><RegisterForm /></BrowserRouter>);

const submit = () => fireEvent.click(screen.getByRole('button', { name: /sign up/i }));

describe('RegisterForm client-side validation', () => {
  test('shows "Name is required!" when the name field is left blank', () => {
    renderForm();

    // Name left blank; fill the rest so only the name check is in play.
    fireEvent.change(screen.getByPlaceholderText('Email'), { target: { value: 'a@example.com' } });
    fireEvent.change(screen.getByPlaceholderText('Password'), { target: { value: 'Str0ng&Pass1' } });
    fireEvent.change(screen.getByPlaceholderText('Confirm Password'), { target: { value: 'Str0ng&Pass1' } });
    submit();

    expect(screen.getByText('Name is required!')).toBeInTheDocument();
    expect(screen.queryByText('Valid name is required!')).not.toBeInTheDocument();
  });

  test('shows "Valid name is required!" for a non-empty but invalid name', () => {
    renderForm();

    fireEvent.change(screen.getByPlaceholderText('Name'), { target: { value: 'John123' } });
    fireEvent.change(screen.getByPlaceholderText('Email'), { target: { value: 'a@example.com' } });
    fireEvent.change(screen.getByPlaceholderText('Password'), { target: { value: 'Str0ng&Pass1' } });
    fireEvent.change(screen.getByPlaceholderText('Confirm Password'), { target: { value: 'Str0ng&Pass1' } });
    submit();

    expect(screen.getByText('Valid name is required!')).toBeInTheDocument();
  });

  test('shows no name error for a valid name', () => {
    // This case passes client-side validation, so submitHandler actually
    // fires the registration request — give it a resolved response so the
    // async .then() chain in storeDataHandler doesn't reject on `undefined`.
    axiosInstance.post.mockResolvedValueOnce({ status: 201, data: {} });
    renderForm();

    fireEvent.change(screen.getByPlaceholderText('Name'), { target: { value: 'John Smith' } });
    fireEvent.change(screen.getByPlaceholderText('Email'), { target: { value: 'a@example.com' } });
    fireEvent.change(screen.getByPlaceholderText('Password'), { target: { value: 'Str0ng&Pass1' } });
    fireEvent.change(screen.getByPlaceholderText('Confirm Password'), { target: { value: 'Str0ng&Pass1' } });
    submit();

    expect(screen.queryByText('Name is required!')).not.toBeInTheDocument();
    expect(screen.queryByText('Valid name is required!')).not.toBeInTheDocument();
  });
});
