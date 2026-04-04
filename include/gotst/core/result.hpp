#pragma once

#include <string>
#include <variant>

namespace gotst {

enum class ErrorCode : int {
    Ok = 0,
    InvalidArgument = 1,
    InvalidState = 2,
    NotFound = 3,
    IoError = 4,
    ModelNotLoaded = 5,
    InferenceFailed = 6,
    ShapeMismatch = 7,
    EmptyInput = 8,
    Cancelled = 9,
};

struct Error {
    ErrorCode code = ErrorCode::Ok;
    std::string message;

    static Error ok() { return {ErrorCode::Ok, {}}; }
    static Error invalid_argument(const std::string &msg) { return {ErrorCode::InvalidArgument, msg}; }
    static Error invalid_state(const std::string &msg) { return {ErrorCode::InvalidState, msg}; }
    static Error not_found(const std::string &msg) { return {ErrorCode::NotFound, msg}; }
    static Error io_error(const std::string &msg) { return {ErrorCode::IoError, msg}; }
    static Error model_not_loaded(const std::string &msg) { return {ErrorCode::ModelNotLoaded, msg}; }
    static Error inference_failed(const std::string &msg) { return {ErrorCode::InferenceFailed, msg}; }
    static Error shape_mismatch(const std::string &msg) { return {ErrorCode::ShapeMismatch, msg}; }
    static Error empty_input(const std::string &msg) { return {ErrorCode::EmptyInput, msg}; }
    static Error cancelled(const std::string &msg) { return {ErrorCode::Cancelled, msg}; }

    bool is_ok() const { return code == ErrorCode::Ok; }
    explicit operator bool() const { return is_ok(); }
};

template <typename T>
class Result {
public:
    Result(T value) : storage_(std::in_place_index<0>, std::move(value)) {}
    Result(Error error) : storage_(std::in_place_index<1>, std::move(error)) {}

    bool is_ok() const { return storage_.index() == 0; }
    explicit operator bool() const { return is_ok(); }

    const T &value() const { return std::get<0>(storage_); }
    T &value() { return std::get<0>(storage_); }

    const Error &get_error() const {
        return is_ok() ? ok_singleton : std::get<1>(storage_);
    }
    ErrorCode error_code() const { return get_error().code; }
    const std::string &error_message() const { return get_error().message; }

private:
    static inline Error ok_singleton = Error::ok();
    std::variant<T, Error> storage_;
};

template <>
class Result<void> {
public:
    Result() : error_(Error::ok()) {}
    Result(Error error) : error_(std::move(error)) {}

    bool is_ok() const { return error_.is_ok(); }
    explicit operator bool() const { return is_ok(); }

    const Error &get_error() const { return error_; }
    ErrorCode error_code() const { return error_.code; }
    const std::string &error_message() const { return error_.message; }

private:
    Error error_;
};

} // namespace gotst
