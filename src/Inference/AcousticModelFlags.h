#ifndef DS_ONNX_INFER_ACOUSTICMODELFLAGS_H
#define DS_ONNX_INFER_ACOUSTICMODELFLAGS_H


namespace diffsinger {
    class AcousticModelFlags {
    public:
        enum Flags : unsigned int {
            Valid = 1 << 0,
            Velocity = 1 << 1,
            Gender = 1 << 2,
            MultiSpeakers = 1 << 3,
            Energy = 1 << 4,
            Breathiness = 1 << 5,
            ShallowDiffusion = 1 << 6
        };
    private:
        unsigned int m_flag;
    public:
        constexpr AcousticModelFlags();

        constexpr void set(Flags flag);
        constexpr void setIf(Flags flag, bool condition);
        constexpr void unset(Flags flag);
        constexpr void toggle(Flags flag);
        constexpr void reset();
        constexpr bool check(Flags flag) const;
    };

    constexpr AcousticModelFlags::AcousticModelFlags() : m_flag(0) {}

    constexpr void AcousticModelFlags::set(AcousticModelFlags::Flags flag) {
        m_flag |= flag;
    }

    constexpr void AcousticModelFlags::setIf(AcousticModelFlags::Flags flag, bool condition) {
        if (condition) {
            set(flag);
        } else {
            unset(flag);
        }
    }

    constexpr void AcousticModelFlags::unset(AcousticModelFlags::Flags flag) {
        m_flag &= ~flag;
    }

    constexpr void AcousticModelFlags::toggle(AcousticModelFlags::Flags flag) {
        m_flag ^= flag;
    }

    constexpr bool AcousticModelFlags::check(AcousticModelFlags::Flags flag) const {
        return m_flag & flag;
    }

    constexpr void AcousticModelFlags::reset() {
        m_flag = 0;
    }
}

#endif //DS_ONNX_INFER_ACOUSTICMODELFLAGS_H
