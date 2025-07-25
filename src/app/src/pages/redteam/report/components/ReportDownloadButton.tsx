import React, { useState } from 'react';

import { useTelemetry } from '@app/hooks/useTelemetry';
import DownloadIcon from '@mui/icons-material/Download';
import CircularProgress from '@mui/material/CircularProgress';
import IconButton from '@mui/material/IconButton';
import Menu from '@mui/material/Menu';
import MenuItem from '@mui/material/MenuItem';
import Tooltip from '@mui/material/Tooltip';
import { convertEvalDataToCsv } from '../utils/csvExport';
import type { ResultsFile } from '@promptfoo/types';

interface ReportDownloadButtonProps {
  evalDescription: string;
  evalData: ResultsFile;
}

const ReportDownloadButton: React.FC<ReportDownloadButtonProps> = ({
  evalDescription,
  evalData,
}) => {
  const [isDownloading, setIsDownloading] = useState(false);
  const [isHovering, setIsHovering] = useState(false);
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);

  const { recordEvent } = useTelemetry();

  const handleClick = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleClose = () => {
    setAnchorEl(null);
  };

  const getFilename = (extension: string) => {
    return evalDescription
      ? `report_${evalDescription
          .toLowerCase()
          .replace(/[^a-z0-9]+/g, '-')
          .replace(/(^-|-$)/g, '')}.${extension}`
      : `report.${extension}`;
  };

  const handleCsvDownload = () => {
    setIsDownloading(true);
    handleClose();

    // Track report export
    recordEvent('webui_action', {
      action: 'redteam_report_export',
      format: 'csv',
    });

    try {
      const csv = convertEvalDataToCsv(evalData);
      const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
      const link = document.createElement('a');
      if (link.download !== undefined) {
        const url = URL.createObjectURL(blob);
        link.setAttribute('href', url);
        link.setAttribute('download', getFilename('csv'));
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
      }
    } catch (error) {
      console.error('Error generating CSV:', error);
    } finally {
      setIsDownloading(false);
    }
  };

  const handleJsonDownload = () => {
    setIsDownloading(true);
    handleClose();

    // Track report export
    recordEvent('webui_action', {
      action: 'redteam_report_export',
      format: 'json',
    });

    try {
      const jsonData = JSON.stringify(evalData, null, 2);
      const blob = new Blob([jsonData], { type: 'application/json;charset=utf-8;' });
      const link = document.createElement('a');
      if (link.download !== undefined) {
        const url = URL.createObjectURL(blob);
        link.setAttribute('href', url);
        link.setAttribute('download', getFilename('json'));
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
      }
    } catch (error) {
      console.error('Error generating JSON:', error);
    } finally {
      setIsDownloading(false);
    }
  };
  const handlePdfDownload = () => {
    handleClose();
    // Track report export
    recordEvent('webui_action', {
      action: 'redteam_report_export',
      format: 'pdf',
    });
    window.print();
  };

  return (
    <>
      <Tooltip title="Download report" placement="top" open={isHovering && !isDownloading}>
        <IconButton
          onClick={handleClick}
          onMouseEnter={() => setIsHovering(true)}
          onMouseLeave={() => setIsHovering(false)}
          sx={{ mt: '4px', position: 'relative' }}
          aria-label="download report"
          disabled={isDownloading}
        >
          {isDownloading ? <CircularProgress size={20} /> : <DownloadIcon />}
        </IconButton>
      </Tooltip>
      <Menu
        anchorEl={anchorEl}
        open={Boolean(anchorEl)}
        onClose={handleClose}
        anchorOrigin={{
          vertical: 'bottom',
          horizontal: 'right',
        }}
        transformOrigin={{
          vertical: 'top',
          horizontal: 'right',
        }}
      >
        <MenuItem onClick={handlePdfDownload}>PDF</MenuItem>
        <MenuItem onClick={handleCsvDownload}>CSV</MenuItem>
        <MenuItem onClick={handleJsonDownload}>JSON</MenuItem>
      </Menu>
    </>
  );
};

export default ReportDownloadButton;
