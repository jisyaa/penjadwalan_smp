-- phpMyAdmin SQL Dump
-- version 5.2.1
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Generation Time: Dec 09, 2025 at 09:01 AM
-- Server version: 10.4.32-MariaDB
-- PHP Version: 8.2.12

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `db_penjadwalan`
--

-- --------------------------------------------------------

--
-- Table structure for table `guru`
--

CREATE TABLE `guru` (
  `id_guru` int(11) NOT NULL,
  `nama_guru` varchar(100) NOT NULL,
  `nip` varchar(20) DEFAULT NULL,
  `jam_mingguan` int(11) NOT NULL,
  `mapel` varchar(100) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `guru`
--

INSERT INTO `guru` (`id_guru`, `nama_guru`, `nip`, `jam_mingguan`, `mapel`) VALUES
(1, 'Afriwarman', '196512071995121001', 30, 'Matematika'),
(2, 'Akmaluddin Mulis', '197402202005011005', 12, 'PJOK'),
(3, 'Delviani Surya', '198807202023212035', 24, 'Bahasa Indonesia'),
(4, 'Desriyati', '199002122023212032', 17, 'IPA, Informatika, Matematika'),
(5, 'Efi Syuryani', '196712031998022001', 20, 'IPA'),
(6, 'Eli Topia', '198406092014062009', 28, 'IPS'),
(7, 'Faredna. Z', '199201052025212026', 24, 'Informatika'),
(8, 'First Putri Yandi', '199307182023212027', 27, 'BK'),
(9, 'Gusma Suci Ramadhani', '199403022023212025', 24, 'Bahasa Indonesia'),
(10, 'Harni Yetti', '196704051991032005', 15, 'IPA'),
(11, 'Husnijar Anshariah', '196801161991032005', 20, 'IPA'),
(12, 'Irdanely', '196605091989032004', 24, 'Bahasa Inggris'),
(13, 'IRWAN WAHYUDI', NULL, 16, 'Bahasa Inggris'),
(14, 'Ivanny Saktya Octorina', NULL, 18, 'Bahasa Indonesia'),
(15, 'Lamimi Agus', NULL, 27, 'Seni Budaya dan Prakarya'),
(16, 'Lendriati', '197005012005012007', 12, 'Bahasa Indonesia'),
(17, 'Leni Marlina M', '197707302005012005', 24, 'PAI'),
(18, 'Levana Ariani', '198110072006042007', 15, 'IPA'),
(19, 'LISA SUSANTI', NULL, 21, 'PAI'),
(20, 'M.Iqbal', NULL, 21, 'PJOK'),
(21, 'Marlis', '196608062008012003', 24, 'IPS'),
(22, 'Marningsih', '197603262010012005', 21, 'Informatika'),
(23, 'MUTHIA GUSTIANDI', '199608182023212015', 24, 'PJOK'),
(24, 'Nadila', NULL, 18, 'BK'),
(25, 'Nurdini', '196608252000122002', 24, 'IPA'),
(26, 'PETRI MELDA DIANI', NULL, 27, 'Pendidikan Pancasila'),
(27, 'Sastika Randra', NULL, 24, 'Bahasa Indonesia'),
(28, 'Sri Ayu Ramadhani', NULL, 24, 'Seni Budaya dan Prakarya'),
(29, 'Sumarni', '196701111989032001', 30, 'Matematika'),
(30, 'Sylvia Eliza Azwar', '198809242017082002', 24, 'Bahasa Indonesia'),
(31, 'Tati Tisnawati', '198205112005012016', 15, 'Matematika'),
(32, 'WAIDIS', '198812312023211015', 24, 'PAI'),
(33, 'Waslul Abral', '196905041997021003', 24, 'IPS'),
(34, 'Yunimar', '196608262008012002', 30, 'Pendidikan Pancasila'),
(35, 'Yusrita', '197506292005012006', 24, 'Bahasa Inggris');

-- --------------------------------------------------------

--
-- Table structure for table `guru_mapel`
--

CREATE TABLE `guru_mapel` (
  `id` int(11) NOT NULL,
  `id_guru` int(11) DEFAULT NULL,
  `id_mapel` int(11) DEFAULT NULL,
  `id_kelas` int(11) DEFAULT NULL,
  `aktif` enum('aktif','tidak') NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `guru_mapel`
--

INSERT INTO `guru_mapel` (`id`, `id_guru`, `id_mapel`, `id_kelas`, `aktif`) VALUES
(1, 31, 1, 14, 'aktif'),
(2, 30, 2, 14, 'aktif'),
(3, 25, 3, 14, 'aktif'),
(4, 33, 4, 14, 'aktif'),
(5, 12, 5, 14, 'aktif'),
(6, 32, 6, 14, 'aktif'),
(7, 34, 7, 14, 'aktif'),
(8, 2, 8, 14, 'aktif'),
(9, 15, 9, 14, 'aktif'),
(10, 7, 10, 14, 'aktif'),
(11, 31, 1, 15, 'aktif'),
(12, 30, 2, 15, 'aktif'),
(13, 25, 3, 15, 'aktif'),
(14, 33, 4, 15, 'aktif'),
(15, 12, 5, 15, 'aktif'),
(16, 32, 6, 15, 'aktif'),
(17, 34, 7, 15, 'aktif'),
(18, 2, 8, 15, 'aktif'),
(19, 15, 9, 15, 'aktif'),
(20, 7, 10, 15, 'aktif'),
(21, 31, 1, 16, 'aktif'),
(22, 30, 2, 16, 'aktif'),
(23, 25, 3, 16, 'aktif'),
(24, 33, 4, 16, 'aktif'),
(25, 12, 5, 16, 'aktif'),
(26, 32, 6, 16, 'aktif'),
(27, 34, 7, 16, 'aktif'),
(28, 2, 8, 16, 'aktif'),
(29, 15, 9, 16, 'aktif'),
(30, 7, 10, 16, 'aktif'),
(31, 20, 1, 17, 'aktif'),
(32, 30, 2, 17, 'aktif'),
(33, 25, 3, 17, 'aktif'),
(34, 33, 4, 17, 'aktif'),
(35, 12, 5, 17, 'aktif'),
(36, 17, 6, 17, 'aktif'),
(37, 34, 7, 17, 'aktif'),
(38, 23, 8, 17, 'aktif'),
(39, 15, 9, 17, 'aktif'),
(40, 7, 10, 17, 'aktif'),
(41, 20, 1, 18, 'aktif'),
(42, 14, 2, 18, 'aktif'),
(43, 5, 3, 18, 'aktif'),
(44, 33, 4, 18, 'aktif'),
(45, 12, 5, 18, 'aktif'),
(46, 17, 6, 18, 'aktif'),
(47, 34, 7, 18, 'aktif'),
(48, 23, 8, 18, 'aktif'),
(49, 15, 9, 18, 'aktif'),
(50, 7, 10, 18, 'aktif'),
(51, 1, 1, 19, 'aktif'),
(52, 14, 2, 19, 'aktif'),
(53, 5, 3, 19, 'aktif'),
(54, 33, 4, 19, 'aktif'),
(55, 12, 5, 19, 'aktif'),
(56, 17, 6, 19, 'aktif'),
(57, 34, 7, 19, 'aktif'),
(58, 23, 8, 19, 'aktif'),
(59, 15, 9, 19, 'aktif'),
(60, 7, 10, 19, 'aktif'),
(61, 1, 1, 7, 'aktif'),
(62, 9, 2, 7, 'aktif'),
(63, 10, 3, 7, 'aktif'),
(64, 6, 4, 7, 'aktif'),
(65, 35, 5, 7, 'aktif'),
(66, 17, 6, 7, 'aktif'),
(67, 26, 7, 7, 'aktif'),
(68, 20, 8, 7, 'aktif'),
(69, 28, 9, 7, 'aktif'),
(70, 7, 10, 7, 'aktif'),
(71, 1, 1, 8, 'aktif'),
(72, 9, 2, 8, 'aktif'),
(73, 10, 3, 8, 'aktif'),
(74, 6, 4, 8, 'aktif'),
(75, 35, 5, 8, 'aktif'),
(76, 17, 6, 8, 'aktif'),
(77, 26, 7, 8, 'aktif'),
(78, 20, 8, 8, 'aktif'),
(79, 28, 9, 8, 'aktif'),
(80, 7, 10, 8, 'aktif'),
(81, 4, 1, 9, 'aktif'),
(82, 9, 2, 9, 'aktif'),
(83, 10, 3, 9, 'aktif'),
(84, 6, 4, 9, 'aktif'),
(85, 35, 5, 9, 'aktif'),
(86, 17, 6, 9, 'aktif'),
(87, 26, 7, 9, 'aktif'),
(88, 20, 8, 9, 'aktif'),
(89, 14, 9, 9, 'aktif'),
(90, 22, 10, 9, 'aktif'),
(91, 22, 1, 10, 'aktif'),
(92, 9, 2, 10, 'aktif'),
(93, 11, 3, 10, 'aktif'),
(94, 6, 4, 10, 'aktif'),
(95, 35, 5, 10, 'aktif'),
(96, 17, 6, 10, 'aktif'),
(97, 26, 7, 10, 'aktif'),
(98, 20, 8, 10, 'aktif'),
(99, 14, 9, 10, 'aktif'),
(100, 4, 10, 10, 'aktif'),
(101, 1, 1, 11, 'aktif'),
(102, 3, 2, 11, 'aktif'),
(103, 11, 3, 11, 'aktif'),
(104, 6, 4, 11, 'aktif'),
(105, 35, 5, 11, 'aktif'),
(106, 19, 6, 11, 'aktif'),
(107, 26, 7, 11, 'aktif'),
(108, 20, 8, 11, 'aktif'),
(109, 15, 9, 11, 'aktif'),
(110, 4, 10, 11, 'aktif'),
(111, 1, 1, 12, 'aktif'),
(112, 9, 2, 12, 'aktif'),
(113, 11, 3, 12, 'aktif'),
(114, 6, 4, 12, 'aktif'),
(115, 35, 5, 12, 'aktif'),
(116, 19, 6, 12, 'aktif'),
(117, 26, 7, 12, 'aktif'),
(118, 20, 8, 12, 'aktif'),
(119, 15, 9, 12, 'aktif'),
(120, 4, 10, 12, 'aktif'),
(121, 1, 1, 13, 'aktif'),
(122, 14, 2, 13, 'aktif'),
(123, 5, 3, 13, 'aktif'),
(124, 6, 4, 13, 'aktif'),
(125, 13, 5, 13, 'aktif'),
(126, 19, 6, 13, 'aktif'),
(127, 26, 7, 13, 'aktif'),
(128, 20, 8, 13, 'aktif'),
(129, 15, 9, 13, 'aktif'),
(130, 4, 10, 13, 'aktif'),
(131, 29, 1, 1, 'aktif'),
(132, 27, 2, 1, 'aktif'),
(133, 18, 3, 1, 'aktif'),
(134, 21, 4, 1, 'aktif'),
(135, 13, 5, 1, 'aktif'),
(136, 32, 6, 1, 'aktif'),
(137, 34, 7, 1, 'aktif'),
(138, 23, 8, 1, 'aktif'),
(139, 28, 9, 1, 'aktif'),
(140, 22, 10, 1, 'aktif'),
(141, 29, 1, 2, 'aktif'),
(142, 27, 2, 2, 'aktif'),
(143, 18, 3, 2, 'aktif'),
(144, 21, 4, 2, 'aktif'),
(145, 13, 5, 2, 'aktif'),
(146, 32, 6, 2, 'aktif'),
(147, 34, 7, 2, 'aktif'),
(148, 23, 8, 2, 'aktif'),
(149, 28, 9, 2, 'aktif'),
(150, 22, 10, 2, 'aktif'),
(151, 29, 1, 3, 'aktif'),
(152, 27, 2, 3, 'aktif'),
(153, 18, 3, 3, 'aktif'),
(154, 21, 4, 3, 'aktif'),
(155, 13, 5, 3, 'aktif'),
(156, 32, 6, 3, 'aktif'),
(157, 34, 7, 3, 'aktif'),
(158, 23, 8, 3, 'aktif'),
(159, 28, 9, 3, 'aktif'),
(160, 22, 10, 3, 'aktif'),
(161, 29, 1, 4, 'aktif'),
(162, 27, 2, 4, 'aktif'),
(163, 5, 3, 4, 'aktif'),
(164, 21, 4, 4, 'aktif'),
(165, 16, 5, 4, 'aktif'),
(166, 32, 6, 4, 'aktif'),
(167, 34, 7, 4, 'aktif'),
(168, 23, 8, 4, 'aktif'),
(169, 28, 9, 4, 'aktif'),
(170, 22, 10, 4, 'aktif'),
(171, 29, 1, 5, 'aktif'),
(172, 27, 2, 5, 'aktif'),
(173, 5, 3, 5, 'aktif'),
(174, 21, 4, 5, 'aktif'),
(175, 16, 5, 5, 'aktif'),
(176, 32, 6, 5, 'aktif'),
(177, 34, 7, 5, 'aktif'),
(178, 23, 8, 5, 'aktif'),
(179, 28, 9, 5, 'aktif'),
(180, 22, 10, 5, 'aktif'),
(181, 29, 1, 6, 'aktif'),
(182, 27, 2, 6, 'aktif'),
(183, 4, 3, 6, 'aktif'),
(184, 21, 4, 6, 'aktif'),
(185, 16, 5, 6, 'aktif'),
(186, 32, 6, 6, 'aktif'),
(187, 34, 7, 6, 'aktif'),
(188, 23, 8, 6, 'aktif'),
(189, 28, 9, 6, 'aktif'),
(190, 22, 10, 6, 'aktif');

-- --------------------------------------------------------

--
-- Table structure for table `jadwal`
--

CREATE TABLE `jadwal` (
  `id_jadwal` int(11) NOT NULL,
  `id_master` int(11) DEFAULT NULL,
  `id_kelas` int(11) DEFAULT NULL,
  `id_mapel` int(11) DEFAULT NULL,
  `id_guru` int(11) DEFAULT NULL,
  `id_ruang` int(11) DEFAULT NULL,
  `id_waktu` int(11) DEFAULT NULL,
  `generasi` int(11) DEFAULT NULL,
  `fitness` float DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `jadwal_master`
--

CREATE TABLE `jadwal_master` (
  `id_master` int(11) NOT NULL,
  `tahun_ajaran` varchar(20) DEFAULT NULL,
  `semester` enum('ganjil','genap') DEFAULT NULL,
  `keterangan` text DEFAULT NULL,
  `dibuat_pada` timestamp NOT NULL DEFAULT current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `kelas`
--

CREATE TABLE `kelas` (
  `id_kelas` int(11) NOT NULL,
  `nama_kelas` varchar(20) NOT NULL,
  `tingkat` enum('7','8','9') NOT NULL,
  `jumlah_siswa` int(11) NOT NULL,
  `wali_kelas` int(11) NOT NULL,
  `id_ruang` int(11) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `kelas`
--

INSERT INTO `kelas` (`id_kelas`, `nama_kelas`, `tingkat`, `jumlah_siswa`, `wali_kelas`, `id_ruang`) VALUES
(1, 'VII A', '7', 31, 29, 12),
(2, 'VII B', '7', 32, 8, 13),
(3, 'VII C', '7', 32, 32, 14),
(4, 'VII D', '7', 28, 5, 15),
(5, 'VII E', '7', 27, 21, 16),
(6, 'VII F', '7', 25, 22, 17),
(7, 'VIII A', '8', 32, 10, 18),
(8, 'VIII B', '8', 31, 7, 19),
(9, 'VIII C', '8', 30, 35, 20),
(10, 'VIII D', '8', 27, 9, 21),
(11, 'VIII E', '8', 27, 3, 22),
(12, 'VIII F', '8', 26, 6, 23),
(13, 'VIII G', '8', 25, 4, 24),
(14, 'IX A', '9', 32, 34, 25),
(15, 'IX B', '9', 32, 30, 26),
(16, 'IX C', '9', 32, 25, 27),
(17, 'IX D', '9', 31, 12, 28),
(18, 'IX E', '9', 25, 23, 29),
(19, 'IX F', '9', 25, 17, 30);

-- --------------------------------------------------------

--
-- Table structure for table `konfigurasi_ag`
--

CREATE TABLE `konfigurasi_ag` (
  `id` int(11) NOT NULL,
  `ukuran_populasi` int(11) DEFAULT 50,
  `probabilitas_crossover` float DEFAULT 0.8,
  `probabilitas_mutasi` float DEFAULT 0.1,
  `jumlah_generasi` int(11) DEFAULT 100
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `log_conflict`
--

CREATE TABLE `log_conflict` (
  `id` int(11) NOT NULL,
  `id_kelas` int(11) DEFAULT NULL,
  `id_guru` int(11) DEFAULT NULL,
  `id_ruang` int(11) DEFAULT NULL,
  `id_waktu` int(11) DEFAULT NULL,
  `jenis_konflik` enum('guru_bentrok','ruang_bentrok','kelas_bentrok') NOT NULL,
  `generasi` int(11) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `mapel`
--

CREATE TABLE `mapel` (
  `id_mapel` int(11) NOT NULL,
  `nama_mapel` varchar(100) NOT NULL,
  `jam_per_minggu` int(11) NOT NULL,
  `kategori` varchar(20) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `mapel`
--

INSERT INTO `mapel` (`id_mapel`, `nama_mapel`, `jam_per_minggu`, `kategori`) VALUES
(1, 'Matematika', 5, 'Teori'),
(2, 'Bahasa Indonesia', 6, 'Teori'),
(3, 'IPA', 5, 'Teori'),
(4, 'IPS', 4, 'Teori'),
(5, 'Bahasa Inggris', 4, 'Teori'),
(6, 'PAI', 3, 'Teori'),
(7, 'Pendidikan Pancasila', 3, 'Teori'),
(8, 'PJOK', 3, 'Praktek'),
(9, 'Seni Budaya dan Prakarya', 3, 'Teori'),
(10, 'Informatika', 3, 'Teori');

-- --------------------------------------------------------

--
-- Table structure for table `ruang`
--

CREATE TABLE `ruang` (
  `id_ruang` int(11) NOT NULL,
  `nama_ruang` varchar(50) NOT NULL,
  `tipe` enum('kelas','laboratorium','ruangan') DEFAULT 'kelas',
  `kapasitas` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `ruang`
--

INSERT INTO `ruang` (`id_ruang`, `nama_ruang`, `tipe`, `kapasitas`) VALUES
(1, 'Ruang TU dan Kepsek', 'ruangan', 35),
(2, 'Lab Komputer 1', 'laboratorium', 40),
(3, 'Lab Komputer 2', 'laboratorium', 40),
(4, 'Ruang Majelis Guru', 'ruangan', 50),
(5, 'Koperasi', 'kelas', 35),
(6, 'Lab IPA', 'laboratorium', 50),
(7, 'Ruang BK', 'ruangan', 35),
(8, 'Ruang Olahraga', 'ruangan', 35),
(9, 'Ruang Pramuka', 'ruangan', 35),
(10, 'Ruang Kesenian', 'ruangan', 35),
(11, 'Ruang UKS', 'ruangan', 35),
(12, 'Kelas VII A', 'kelas', 35),
(13, 'Kelas VII B', 'kelas', 35),
(14, 'Kelas VII C', 'kelas', 35),
(15, 'Kelas VII D', 'kelas', 35),
(16, 'Kelas VII E', 'kelas', 35),
(17, 'Kelas VII F', 'kelas', 35),
(18, 'Kelas VIII A', 'kelas', 35),
(19, 'Kelas VIII B', 'kelas', 35),
(20, 'Kelas VIII C', 'kelas', 35),
(21, 'Kelas VIII D', 'kelas', 35),
(22, 'Kelas VIII E', 'kelas', 35),
(23, 'Kelas VIII F', 'kelas', 35),
(24, 'Kelas VIII G', 'kelas', 35),
(25, 'Kelas IX A', 'kelas', 35),
(26, 'Kelas IX B', 'kelas', 35),
(27, 'Kelas IX C', 'kelas', 35),
(28, 'Kelas IX D', 'kelas', 35),
(29, 'Kelas IX E', 'kelas', 35),
(30, 'Kelas IX F', 'kelas', 35);

-- --------------------------------------------------------

--
-- Table structure for table `waktu`
--

CREATE TABLE `waktu` (
  `id_waktu` int(11) NOT NULL,
  `hari` enum('Senin','Selasa','Rabu','Kamis','Jumat') NOT NULL,
  `jam_ke` int(11) DEFAULT NULL,
  `waktu_mulai` time DEFAULT NULL,
  `waktu_selesai` time DEFAULT NULL,
  `keterangan` varchar(50) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `waktu`
--

INSERT INTO `waktu` (`id_waktu`, `hari`, `jam_ke`, `waktu_mulai`, `waktu_selesai`, `keterangan`) VALUES
(1, 'Senin', 1, '07:15:00', '08:10:00', 'Upacara'),
(2, 'Senin', 2, '08:10:00', '08:50:00', ''),
(3, 'Senin', 3, '08:50:00', '09:30:00', ''),
(4, 'Senin', 4, '09:30:00', '10:10:00', ''),
(5, 'Senin', NULL, '10:10:00', '10:30:00', 'Istirahat'),
(6, 'Senin', 5, '10:30:00', '11:10:00', ''),
(7, 'Senin', 6, '11:10:00', '11:50:00', ''),
(8, 'Senin', 7, '11:50:00', '12:30:00', ''),
(9, 'Senin', NULL, '12:30:00', '13:10:00', 'Ishoma'),
(10, 'Senin', 8, '13:10:00', '13:50:00', ''),
(11, 'Senin', 9, '13:50:00', '14:30:00', ''),
(12, 'Senin', 10, '14:30:00', '15:10:00', ''),
(13, 'Selasa', 1, '07:15:00', '08:10:00', 'Literasi'),
(14, 'Selasa', 2, '08:10:00', '08:50:00', ''),
(15, 'Selasa', 3, '08:50:00', '09:30:00', ''),
(16, 'Selasa', 4, '09:30:00', '10:10:00', ''),
(17, 'Selasa', NULL, '10:10:00', '10:30:00', 'Istirahat'),
(18, 'Selasa', 5, '10:30:00', '11:10:00', ''),
(19, 'Selasa', 6, '11:10:00', '11:50:00', ''),
(20, 'Selasa', 7, '11:50:00', '12:30:00', ''),
(21, 'Selasa', NULL, '12:30:00', '13:10:00', 'Ishoma'),
(22, 'Selasa', 8, '13:10:00', '13:50:00', ''),
(23, 'Selasa', 9, '13:50:00', '14:30:00', ''),
(24, 'Selasa', 10, '14:30:00', '16:00:00', 'Ekstrakulikuler'),
(25, 'Rabu', 1, '07:15:00', '08:10:00', 'Tahfidz'),
(26, 'Rabu', 2, '08:10:00', '08:50:00', ''),
(27, 'Rabu', 3, '08:50:00', '09:30:00', ''),
(28, 'Rabu', 4, '09:30:00', '10:10:00', ''),
(29, 'Rabu', NULL, '10:10:00', '10:30:00', 'Istirahat'),
(30, 'Rabu', 5, '10:30:00', '11:10:00', ''),
(31, 'Rabu', 6, '11:10:00', '11:50:00', ''),
(32, 'Rabu', 7, '11:50:00', '12:30:00', ''),
(33, 'Rabu', NULL, '12:30:00', '13:10:00', 'Ishoma'),
(34, 'Rabu', 8, '13:10:00', '13:50:00', ''),
(35, 'Rabu', 9, '13:50:00', '14:30:00', ''),
(36, 'Rabu', 10, '14:30:00', '16:00:00', 'Ekstrakulikuler'),
(37, 'Kamis', 1, '07:15:00', '08:10:00', 'Literasi'),
(38, 'Kamis', 2, '08:10:00', '08:50:00', ''),
(39, 'Kamis', 3, '08:50:00', '09:30:00', ''),
(40, 'Kamis', 4, '09:30:00', '10:10:00', ''),
(41, 'Kamis', NULL, '10:10:00', '10:30:00', 'Istirahat'),
(42, 'Kamis', 5, '10:30:00', '11:10:00', ''),
(43, 'Kamis', 6, '11:10:00', '11:50:00', ''),
(44, 'Kamis', 7, '11:50:00', '12:30:00', ''),
(45, 'Kamis', NULL, '12:30:00', '13:10:00', 'Ishoma'),
(46, 'Kamis', 8, '13:10:00', '13:50:00', ''),
(47, 'Kamis', 9, '13:50:00', '14:30:00', ''),
(48, 'Kamis', 10, '14:30:00', '16:00:00', 'Ekstrakulikuler'),
(49, 'Jumat', 1, '07:15:00', '08:00:00', 'Muhadharah'),
(50, 'Jumat', 2, '08:00:00', '08:40:00', ''),
(51, 'Jumat', 3, '08:40:00', '09:20:00', ''),
(52, 'Jumat', 4, '09:20:00', '10:00:00', ''),
(53, 'Jumat', NULL, '10:00:00', '10:20:00', 'Istirahat'),
(54, 'Jumat', 5, '10:20:00', '11:00:00', ''),
(55, 'Jumat', 6, '11:00:00', '11:40:00', '');

--
-- Indexes for dumped tables
--

--
-- Indexes for table `guru`
--
ALTER TABLE `guru`
  ADD PRIMARY KEY (`id_guru`),
  ADD UNIQUE KEY `nip` (`nip`);

--
-- Indexes for table `guru_mapel`
--
ALTER TABLE `guru_mapel`
  ADD PRIMARY KEY (`id`),
  ADD KEY `id_guru` (`id_guru`),
  ADD KEY `id_mapel` (`id_mapel`),
  ADD KEY `id_kelas` (`id_kelas`);

--
-- Indexes for table `jadwal`
--
ALTER TABLE `jadwal`
  ADD PRIMARY KEY (`id_jadwal`),
  ADD KEY `id_master` (`id_master`);

--
-- Indexes for table `jadwal_master`
--
ALTER TABLE `jadwal_master`
  ADD PRIMARY KEY (`id_master`);

--
-- Indexes for table `kelas`
--
ALTER TABLE `kelas`
  ADD PRIMARY KEY (`id_kelas`),
  ADD KEY `id_ruang` (`id_ruang`),
  ADD KEY `kelas_ibfk_2` (`wali_kelas`);

--
-- Indexes for table `konfigurasi_ag`
--
ALTER TABLE `konfigurasi_ag`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `log_conflict`
--
ALTER TABLE `log_conflict`
  ADD PRIMARY KEY (`id`),
  ADD KEY `id_kelas` (`id_kelas`),
  ADD KEY `id_guru` (`id_guru`),
  ADD KEY `id_ruang` (`id_ruang`),
  ADD KEY `id_waktu` (`id_waktu`);

--
-- Indexes for table `mapel`
--
ALTER TABLE `mapel`
  ADD PRIMARY KEY (`id_mapel`);

--
-- Indexes for table `ruang`
--
ALTER TABLE `ruang`
  ADD PRIMARY KEY (`id_ruang`);

--
-- Indexes for table `waktu`
--
ALTER TABLE `waktu`
  ADD PRIMARY KEY (`id_waktu`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `guru`
--
ALTER TABLE `guru`
  MODIFY `id_guru` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=36;

--
-- AUTO_INCREMENT for table `guru_mapel`
--
ALTER TABLE `guru_mapel`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=191;

--
-- AUTO_INCREMENT for table `jadwal`
--
ALTER TABLE `jadwal`
  MODIFY `id_jadwal` int(11) NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT for table `jadwal_master`
--
ALTER TABLE `jadwal_master`
  MODIFY `id_master` int(11) NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT for table `kelas`
--
ALTER TABLE `kelas`
  MODIFY `id_kelas` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=20;

--
-- AUTO_INCREMENT for table `konfigurasi_ag`
--
ALTER TABLE `konfigurasi_ag`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT for table `log_conflict`
--
ALTER TABLE `log_conflict`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT for table `mapel`
--
ALTER TABLE `mapel`
  MODIFY `id_mapel` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=11;

--
-- AUTO_INCREMENT for table `ruang`
--
ALTER TABLE `ruang`
  MODIFY `id_ruang` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=31;

--
-- AUTO_INCREMENT for table `waktu`
--
ALTER TABLE `waktu`
  MODIFY `id_waktu` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=56;

--
-- Constraints for dumped tables
--

--
-- Constraints for table `guru_mapel`
--
ALTER TABLE `guru_mapel`
  ADD CONSTRAINT `guru_mapel_ibfk_1` FOREIGN KEY (`id_guru`) REFERENCES `guru` (`id_guru`) ON DELETE CASCADE,
  ADD CONSTRAINT `guru_mapel_ibfk_2` FOREIGN KEY (`id_mapel`) REFERENCES `mapel` (`id_mapel`) ON DELETE CASCADE,
  ADD CONSTRAINT `guru_mapel_ibfk_3` FOREIGN KEY (`id_kelas`) REFERENCES `kelas` (`id_kelas`) ON DELETE CASCADE;

--
-- Constraints for table `jadwal`
--
ALTER TABLE `jadwal`
  ADD CONSTRAINT `jadwal_ibfk_1` FOREIGN KEY (`id_master`) REFERENCES `jadwal_master` (`id_master`);

--
-- Constraints for table `kelas`
--
ALTER TABLE `kelas`
  ADD CONSTRAINT `kelas_ibfk_1` FOREIGN KEY (`id_ruang`) REFERENCES `ruang` (`id_ruang`) ON DELETE CASCADE,
  ADD CONSTRAINT `kelas_ibfk_2` FOREIGN KEY (`wali_kelas`) REFERENCES `guru` (`id_guru`);

--
-- Constraints for table `log_conflict`
--
ALTER TABLE `log_conflict`
  ADD CONSTRAINT `log_conflict_ibfk_1` FOREIGN KEY (`id_kelas`) REFERENCES `kelas` (`id_kelas`),
  ADD CONSTRAINT `log_conflict_ibfk_2` FOREIGN KEY (`id_guru`) REFERENCES `guru` (`id_guru`),
  ADD CONSTRAINT `log_conflict_ibfk_3` FOREIGN KEY (`id_ruang`) REFERENCES `ruang` (`id_ruang`),
  ADD CONSTRAINT `log_conflict_ibfk_4` FOREIGN KEY (`id_waktu`) REFERENCES `waktu` (`id_waktu`);
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
