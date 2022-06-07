#pragma once

#include <array>

// ==================================================== BASE TYPES ====================================================

template <typename T, size_t SIZE>
using BaseVec = std::array<T, SIZE>;

template <typename T, size_t WIDTH, size_t HEIGHT>
using BaseMtx = std::array<BaseVec<T, WIDTH>, HEIGHT>;

template <typename T, size_t SIZE>
using BaseMtxSq = std::array<BaseVec<T, SIZE>, SIZE>;

// Matrices are indexed as [y][x].  Matrix[i] is the i'th row.

// ==================================================== Scalar Math ====================================================

template <typename T>
T BinomialCoefficient(T n, T k)
{
	T numerator = (T)1.0f;
	T denominator = (T)1.0f;

	for (int i = 1; i <= k; ++i)
	{
		numerator *= T(n + 1 - i);
		denominator *= T(i);
	}

	return numerator / denominator;
}

template <typename T>
T Clamp(T value, T themin, T themax)
{
	if (value < themin)
		return themin;

	if (value > themax)
		return themax;

	return value;
}

// ==================================================== Vector Math ====================================================

template <typename T, size_t SIZE>
T Dot(const BaseVec<T, SIZE>& A, const BaseVec<T, SIZE>& B)
{
	T sum = (T)0.0f;
	for (size_t i = 0; i < SIZE; ++i)
		sum += A[i] * B[i];
	return sum;
}

// ==================================================== Matrix Math ====================================================

template <typename T, size_t WIDTH, size_t HEIGHT>
constexpr size_t Rows(const BaseMtx<T, WIDTH, HEIGHT>& A)
{
	return HEIGHT;
}

template <typename T, size_t WIDTH, size_t HEIGHT>
constexpr size_t Columns(const BaseMtx<T, WIDTH, HEIGHT>& A)
{
	return WIDTH;
}

template <typename T, size_t WIDTH, size_t HEIGHT>
BaseVec<T, HEIGHT> Column(const BaseMtx<T, WIDTH, HEIGHT>& A, size_t index)
{
	BaseVec<T, HEIGHT> ret;
	for (size_t i = 0; i < HEIGHT; ++i)
		ret[i] = A[i][index];
	return ret;
}

template <typename T, size_t WIDTH, size_t HEIGHT>
BaseVec<T, WIDTH> Row(const BaseMtx<T, WIDTH, HEIGHT>& A, size_t index)
{
	return A[index];
}

template <typename T, size_t WIDTH, size_t HEIGHT>
BaseMtx<T, HEIGHT, WIDTH> Transpose(const BaseMtx<T, WIDTH, HEIGHT>& mtx)
{
	BaseMtx<T, HEIGHT, WIDTH> ret;
	for (size_t i = 0; i < WIDTH; ++i)
		ret[i] = Column(mtx, i);
	return ret;
}

template <typename T, size_t WIDTH, size_t HEIGHT>
BaseVec<T, HEIGHT> Multiply(const BaseMtx<T, WIDTH, HEIGHT>& mtx, const BaseVec<T, WIDTH>& vec)
{
	BaseVec<T, HEIGHT> ret{};

	for (size_t i = 0; i < HEIGHT; ++i)
		ret[i] = Dot(Row(mtx, i), vec);

	return ret;
}

template <typename T, size_t WIDTH, size_t HEIGHT>
BaseVec<T, WIDTH> Multiply(const BaseVec<T, HEIGHT>& vec, const BaseMtx<T, WIDTH, HEIGHT>& mtx)
{
	return Multiply(Transpose(mtx), vec);
}

template <typename T, size_t AWIDTH, size_t AHEIGHT, size_t BWIDTH>
BaseMtx<T, BWIDTH, AHEIGHT> Multiply(const BaseMtx<T, AWIDTH, AHEIGHT>& A, const BaseMtx<T, BWIDTH, AWIDTH>& B)
{
	BaseMtx<T, BWIDTH, AHEIGHT> ret{};

	for (int im = 0; im < BWIDTH; ++im)
		for (int in = 0; in < AHEIGHT; ++in)
			ret[in][im] = Dot(Row(A, in), Column(B, im));

	return ret;
}

// ==================================================== Complex Functions ====================================================

template <typename T, size_t WIDTH, size_t HEIGHT>
void GaussJordanElimination(BaseMtx<T, WIDTH, HEIGHT>& augmentedMtx)
{
	// augmentedMtx is assumed to be an NxN matrix possibly with some "extra bits" of size (M-N, N) on the right side.

	// make each column in the matrix have only a single row with a value in it, and have that value be 1.
	for (int column = 0; column < HEIGHT; ++column)
	{
		// find the row that has the maximum absolute value for this column
		int maxValueRowIndex = column;
		T maxValue = augmentedMtx[column][column];
		for (int row = column + 1; row < HEIGHT; ++row)
		{
			if (abs(augmentedMtx[row][column]) > abs(maxValue))
			{
				maxValue = augmentedMtx[row][column];
				maxValueRowIndex = row;
			}
		}

		// swap rows if we need to
		if (column != maxValueRowIndex)
		{
			for (size_t ix = 0; ix < WIDTH; ++ix)
				std::swap(augmentedMtx[column][ix], augmentedMtx[maxValueRowIndex][ix]);
		}

		// scale this row by the value
		{
			T scale = augmentedMtx[column][column];
			for (size_t ix = 0; ix < WIDTH; ++ix)
				augmentedMtx[column][ix] /= scale;
		}

		// make only this row have a value in it, by adding or subtracting multiples of it from the other rows
		for (size_t iy = 0; iy < HEIGHT; ++iy)
		{
			if (iy == column)
				continue;

			T scale = augmentedMtx[iy][column];
			if (scale == 0.0f)
				continue;

			for (size_t ix = 0; ix < WIDTH; ++ix)
				augmentedMtx[iy][ix] -= augmentedMtx[column][ix] * scale;
		}
	}
}
