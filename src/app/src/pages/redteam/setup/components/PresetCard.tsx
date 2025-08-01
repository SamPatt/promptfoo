import Paper from '@mui/material/Paper';
import { alpha } from '@mui/material/styles';
import Typography from '@mui/material/Typography';

interface PresetCardProps {
  name: string;
  description: string;
  isSelected: boolean;
  onClick: () => void;
}

export default function PresetCard({ name, description, isSelected, onClick }: PresetCardProps) {
  return (
    <Paper
      elevation={2}
      onClick={onClick}
      sx={{
        p: 2.5,
        height: '100%',
        cursor: 'pointer',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'flex-start',
        justifyContent: 'flex-start',
        position: 'relative',
        overflow: 'hidden',
        transition: 'all 0.2s ease-in-out',
        border: (theme) =>
          `1px solid ${isSelected ? theme.palette.primary.main : theme.palette.divider}`,
        bgcolor: (theme) =>
          isSelected ? alpha(theme.palette.primary.main, 0.04) : 'background.paper',
        '&:hover': {
          transform: 'translateY(-2px)',
          boxShadow: (theme) => `0 6px 20px ${alpha(theme.palette.common.black, 0.09)}`,
          bgcolor: (theme) =>
            isSelected
              ? alpha(theme.palette.primary.main, 0.08)
              : alpha(theme.palette.action.hover, 0.04),
          '& .MuiTypography-root': {
            color: 'primary.main',
          },
        },
      }}
    >
      <Typography
        variant="h6"
        sx={{
          fontWeight: 500,
          color: isSelected ? 'primary.main' : 'text.primary',
          transition: 'color 0.2s ease-in-out',
          mb: 1.5,
        }}
      >
        {name}
      </Typography>

      <Typography
        variant="body2"
        color="text.secondary"
        sx={{
          opacity: 0.8,
          lineHeight: 1.5,
        }}
      >
        {description}
      </Typography>
    </Paper>
  );
}
