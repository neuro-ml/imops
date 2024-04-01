// Szudzik’s Pairing Function https://en.wikipedia.org/wiki/Pairing_function#Other_pairing_functions
inline uint64_t elegant_pair(size_t a, size_t b) {
    return static_cast<uint64_t>(b) * b + a;  // implicit cast
}

inline size_t fast_mod(const size_t i, const size_t c) {
    return i >= c ? i % c : i;
}

inline bool on_segment(size_t Ax, size_t Ay, size_t Bx, size_t By, size_t Cx, size_t Cy) {
    return Bx <= std::max(Ax, Cx) and Bx >= std::min(Ax, Cx) and By <= std::max(Ay, Cy) and
           By >= std::min(Ay, Cy);
}

inline int32_t orientation(int64_t Ax, int64_t Ay, int64_t Bx, int64_t By, int64_t Cx, int64_t Cy) {
    int64_t val = (By - Ay) * (Cx - Bx) - (Bx - Ax) * (Cy - By);
    if (val > 0) {
        return 1;
    } else if (val < 0) {
        return 2;
    }
    return 0;
}

inline bool segments_intersection(size_t Ax, size_t Ay, size_t Bx, size_t By, size_t Cx, size_t Cy,
                                  size_t Dx, size_t Dy) {
    auto o1 = orientation(Ax, Ay, Bx, By, Cx, Cy);
    auto o2 = orientation(Ax, Ay, Bx, By, Dx, Dy);
    auto o3 = orientation(Cx, Cy, Dx, Dy, Ax, Ay);
    auto o4 = orientation(Cx, Cy, Dx, Dy, Bx, By);
    if ((o1 != o2) and (o3 != o4)) {
        return true;
    }

    if ((o1 == 0) and on_segment(Ax, Ay, Cx, Cy, Bx, By)) {
        return true;
    }

    if ((o2 == 0) and on_segment(Ax, Ay, Dx, Dy, Bx, By)) {
        return true;
    }

    if ((o3 == 0) and on_segment(Cx, Cy, Ax, Ay, Dx, Dy)) {
        return true;
    }

    if ((o4 == 0) and on_segment(Cx, Cy, Bx, By, Dx, Dy)) {
        return true;
    }

    return false;
}

inline bool point_in_triangle(const std::array<int64_t, 4>& coords_info) {
    if (coords_info[3] < 0) {
        return coords_info[3] <= coords_info[0] and coords_info[0] <= 0 and
               coords_info[3] <= coords_info[1] and coords_info[1] <= 0 and
               coords_info[3] <= coords_info[2] and coords_info[2] <= 0;
    }
    return coords_info[3] >= coords_info[0] and coords_info[0] >= 0 and
           coords_info[3] >= coords_info[1] and coords_info[1] >= 0 and
           coords_info[3] >= coords_info[2] and coords_info[2] >= 0;
}
